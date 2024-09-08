
### Several functions are copied and adapted from https://github.com/DZhaoXd/DT-ST/

from utils_source_free.general_imports import * 


def make_a_deepcopy(net, logging): 
    #Makes a deepcopy of the net, and put it into eval mode
    net_copy = copy.deepcopy(net)
    for l_name, l_module in net_copy.named_modules(): 
            if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                    l_module.eval()
    
    net_copy.eval()
    return net_copy



def validation_performance_pseudo_labels(net, config, test_loader, device, cls_thresh_cuda, list_ignore_classes=[0], running_conf=None, THRESHOLD_BETA=None):
    """
    Function that reports the performance of the pseudo labels.
    """
    net.eval()
    error_seg_head = 0
    pseudo_label_rate_sum = 0.0
    ignore_pseudo_label_rate_sum = 0.0
    
    cm_seg_head = np.zeros((config["nb_classes_inference"],config["nb_classes_inference"]))
    mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"], target_dataset=config["target_dataset_name"])
    summation_matrix = summation_matrix_generator(mapping_information)
    summation_matrix = summation_matrix.to(device)
    with torch.no_grad():
        count_iter = 0
       
        for data in test_loader:

            data = dict_to_device(data, device)
            _, output_seg, _ = net.forward_mapped_learned(data) #SOURCE data
            
            if config["parameter"]["pl_no_mapping"]:
                output_merged = output_seg[:,:,0]
            else:
                output_merged = output_seg[:,:,0]@summation_matrix
            
            #Calculate the pseudo-labels
            output = F.softmax(output_merged, dim=1)
            
            
            ###DT-ST setting
            thresolded_label, _, _ = pseudo_labels_probs(output, running_conf, THRESHOLD_BETA)
            thresolded_label = thresolded_label.detach().cpu().numpy()
            # else:
            #     thresolded_label = label_selection(cls_thresh_cuda, output).cpu().numpy()
            
            #The current prediction
            output_seg_np = prediction_changer(output_seg.cpu().detach(), mapping_information) #Only predicting on available classes
            target_seg_np = data["y"].cpu().numpy().astype(int)
            
            #Only evaluate on the ones which are not mapped to 0 as pseudo-labes --> only evaluating the pseudo-labels
            mask_pseudo_labels = thresolded_label !=0
            output_seg_np_pseudo_label = output_seg_np[mask_pseudo_labels]
            target_seg_np_pseudo_label = target_seg_np[mask_pseudo_labels]
            
            pseudo_label_rate_batch = 1 - (np.sum(mask_pseudo_labels)/mask_pseudo_labels.shape[0])
            pseudo_label_rate_sum += pseudo_label_rate_batch
            
            
            true_labels_pseudo_labels = data["y"].detach()[mask_pseudo_labels]
            mask_ignore_points_in_pseudo_labels = true_labels_pseudo_labels==0
            rate_ignore_points_loss_batch = np.sum(mask_ignore_points_in_pseudo_labels.detach().cpu().numpy()) / np.sum(mask_pseudo_labels)
            ignore_pseudo_label_rate_sum += rate_ignore_points_loss_batch
            
            cm_seg_head_ = confusion_matrix(output_seg_np_pseudo_label.ravel(), target_seg_np_pseudo_label.ravel(), labels=list(range(config["nb_classes_inference"])))
            cm_seg_head += cm_seg_head_
            
            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()
        # point wise scores on training segmentation head
        # test_seg_head_oa = metrics.stats_overall_accuracy(cm_seg_head, ignore_list=list_ignore_classes)
        test_seg_head_maa, accuracy_per_class = metrics.stats_accuracy_per_class(cm_seg_head, ignore_list=list_ignore_classes) #First return value is the mean IoU
        test_seg_head_miou, seg_iou_per_class = metrics.stats_iou_per_class(cm_seg_head, ignore_list=list_ignore_classes) #First return value is the mean IoU
        test_seg_head_loss = error_seg_head / cm_seg_head.sum()

    return_data = { "test_seg_head_miou":test_seg_head_miou,\
            "test_seg_head_maa":test_seg_head_maa, "test_seg_head_loss":test_seg_head_loss, "seg_iou_per_class":seg_iou_per_class,\
                "accuracy_per_class":accuracy_per_class, "cm_seg_head":cm_seg_head, "pseudo_label_rate":pseudo_label_rate_sum/count_iter, "ignore_points_rate":ignore_pseudo_label_rate_sum/count_iter}
    return return_data 


def pseudo_label(net_pseudo_label, config, target_train_loader, device, writer, names_list, i, running_conf, THRESHOLD_BETA):
    
        
    #No threshold recalibration needed
    cls_thresh = np.ones(config["nb_classes"], dtype=np.float32)

    
    cls_thresh_cuda = torch.from_numpy(cls_thresh).to(device)
    return_data_pl = validation_performance_pseudo_labels(net_pseudo_label, config, target_train_loader, device, cls_thresh_cuda, [0], running_conf, THRESHOLD_BETA)
    
    logging.info(f"Pseudo Label mIoU: {return_data_pl['test_seg_head_miou']}")
    writer.add_scalar(f"pseudo_label.seg_mIou", return_data_pl['test_seg_head_miou'], i)
    logging.info(f"Pseudo Label Per class {return_data_pl['seg_iou_per_class']}")
    for q in range(len(names_list)):
        writer.add_scalar(f"pseudo_label.seg_Iou_{names_list[q]}", return_data_pl['seg_iou_per_class'][q], i)
    logging.info(f"Pseudo Label rate: {return_data_pl['pseudo_label_rate']}")
    writer.add_scalar(f"pseudo_label.pseudo_label_rate", return_data_pl['pseudo_label_rate'], i)
    
    logging.info(f"Ignore class rates in pseudo label {return_data_pl['ignore_points_rate']}")
    writer.add_scalar(f"pseudo_label.evaluation.pseudo_ignore_points_rate", return_data_pl['ignore_points_rate'], i)
     
    return cls_thresh_cuda

def pseudo_labels_probs(probs, running_conf, THRESHOLD_BETA, RUN_CONF_UPPER=0.80, ignore_augm=None, discount = True):
    ### From https://github.com/DZhaoXd/DT-ST/blob/main/train_TCR_DTU.py#L94
    """Consider top % pixel w.r.t. each image"""
    ###We consider the whole batch
    
    RUN_CONF_UPPER = RUN_CONF_UPPER
    RUN_CONF_LOWER = 0.20
    
    N,C= probs.size()
    max_conf, max_idx = probs.max(1, keepdim=True) # B,1,H,W, take per example the maximum
   
    probs_peaks = torch.zeros_like(probs)
    probs_peaks.scatter_(1, max_idx, max_conf) # B,C,H,W #Write into the zero array the maximum per example
    
    top_peaks, _ = probs_peaks.view(N,C).max(0) # N,C #Get the top peaks per class for the complete batch --> we assume bs=1
    

    # top_peaks 
    top_peaks *= RUN_CONF_UPPER
    
    
    if discount:
        # discount threshold for long-tail classes
        top_peaks *= (1. - torch.exp(- running_conf / THRESHOLD_BETA))
    
    top_peaks.clamp_(RUN_CONF_LOWER) # in-place --> set to a minimal threshold of 20
    probs_peaks.gt_(top_peaks.view(1,C))

    # ignore if lower than the discounted peaks
    ignore = probs_peaks.sum(1, keepdim=True) != 1

    # thresholding the most confident pixels
    pseudo_labels = max_idx.clone()
    pseudo_labels[ignore] = 0
    pseudo_labels = pseudo_labels.squeeze(1)

    return pseudo_labels, max_conf, max_idx


# refer to https://github.com/visinf/da-sac
def update_running_conf(probs, running_conf, THRESHOLD_BETA, tolerance=1e-8):
    """Maintain the moving class prior"""
    STAT_MOMENTUM = 0.9
    
    N,C= probs.size()
    probs_avg = probs.mean(0).view(C,-1).mean(-1)
    # updating the new records: copy the value
    update_index = probs_avg > tolerance
    new_index = update_index & (running_conf == THRESHOLD_BETA)
    running_conf[new_index] = probs_avg[new_index]

    # use the moving average for the rest (Eq. 2)
    running_conf *= STAT_MOMENTUM
    running_conf += (1 - STAT_MOMENTUM) * probs_avg
    return running_conf

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=1)
    en = -torch.sum(p * torch.log(p + 1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en
    
class WeightEMA(object):
    def __init__(self, params, src_params, alpha):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
            

def evel_stu(config, net, stu_eval_list, device, dict_with_mapping, alpha):
    net.eval()
    
    eval_result = []

    with torch.no_grad():
        for i, (target_data, permute_index) in enumerate(stu_eval_list):
            target_data = dict_to_device(target_data, device)
            _, output, _ = net.forward_mapped_learned(target_data)
            output = F.softmax(output, dim=1)[:,:,0]   
            pred1_rand = permute_index
            #select_point = pred1_rand.shape[0]
            select_point = 100
            pred1 = F.normalize(output[pred1_rand[:select_point]])
            pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
            eval_result.append(pred1_en.item())
    
    
    net.train()
    for l_name, l_module in net.named_modules(): 
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()
            
    return eval_result

def main(config_arguments):
    #Setting the seeds
    torch.manual_seed(1234) 
    random.seed(1234)
    np.random.seed(1234)
    
    # define the logging
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter('runs_eccv/{}'.format(f"{config_arguments['tensorboard_folder']}/{config_arguments['name']}"))
    
    
    if os.path.isfile(config_arguments["resume_path"]): 
        #If the precise checkpoint is given take the config in the same directory
        print("File exists")
        file_path_config = os.path.join(os.path.dirname(config_arguments["resume_path"]), "config.yaml")
    else: 
        file_path_config = os.path.join(config_arguments["resume_path"], "config.yaml")

    config = read_yaml_file(file_path_config)
    if config:
    # Access the data in the dictionary
        print(file_path_config)
        print(f"Loaded Config: {config}")    
    else:
        print("Failed to read the config YAML file.")
    logging.getLogger().setLevel(config["logging"])
    config["parameter"] = config_arguments

    ##############################################################################################
    #Selection of the setting that is used

    config = config_adapter(config)

    config["ignore_class"] = 0

    mapping_info = sf_class_mapping_loader(source_dataset=config["source_dataset_name"], target_dataset=config["target_dataset_name"])
    summation_matrix = summation_matrix_generator(mapping_info)
    ##############################################################################################


    ### Iterate over the additional arguments
    for k,v in config_arguments.items():
        logging.info(f"{k}: {v}")


    logging.info("Creating the network")
    logging.info(f"Self-Supervised Setting")
    config['training_batch_size']= config["parameter"]["batch_size"]
    config["test_batch_size"]=16
    savedir_root=f"ckpts_bn/{config_arguments['name']}"
    os.makedirs(savedir_root, exist_ok=True)
    config["ns_dataset_version"] = 'v1.0-trainval'
    config["network_backbone"] = 'TorchSparseMinkUNet_learned'

    name_shift_inverse = {}

    for key,value in name_shift.items():
        name_shift_inverse[value]=key

    config["da_fixed_head_path_model"]=config["parameter"]["resume_path"]

    # device
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True

    bb_dir_root = get_bbdir_root(config)
    # create the network
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]
    decoder = {'name':config["network_decoder"], 'k': config['network_decoder_k']}

    in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
    logging.info("In channels source {}".format(in_channels_source))
    logging.info("in channels target {}".format(in_channels_target))
    logging.info("Creating the network")
    def network_function():
        return networks.Network(in_channels=in_channels_source, latent_size=latent_size, backbone=backbone,\
                    voxel_size=config["voxel_size"], dual_seg_head = config["dual_seg_head"], target_in_channels=in_channels_target, config=config)

    ### Final network
    net_final = network_function()
    ## ckpt_number = -1 means to load the last ckpt
    if os.path.isfile(bb_dir_root): 
        #Load a specified checkpoint 
        ckpt_path=bb_dir_root
        logging.info(f"Failed: Load ckpt from {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    


    #Updating the checkpoint
    checkpoint_new = {}
    for key in checkpoint["state_dict"].keys():
        if key in name_shift_inverse: 
            checkpoint_new[name_shift_inverse[key]] = checkpoint["state_dict"][key]
        else:
            if "num_batches_tracked" in key or "point_transforms" in key: 
                pass
            else: 
                checkpoint_new[key]= checkpoint["state_dict"][key]

    try: 
        net_final.load_state_dict(checkpoint_new)
    except Exception as e: 
        print(e)
        logging.info(f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false")
        net_final.load_state_dict(checkpoint_new, strict=False)

    logging.info(f"Network -- Number of parameters {count_parameters(net_final)}")
    target_DatasetClass = get_dataset(eval("datasets."+config["target_dataset_name"]))
    
    val_number = 1 #1: verifying split, 2 train split, else: test split

    print(f"config dataset source {config}")
    dataloader_dict = da_sf_get_dataloader(target_DatasetClass, config, net_final, network_function, val=val_number, train_shuffle=True, keep_orignal_data=True)

    
    target_train_loader = dataloader_dict ["target_train_loader"]
    target_test_loader = dataloader_dict ["target_test_loader"]

    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))
 
    # create the loss layer
    loss_layer = torch.nn.BCEWithLogitsLoss()
    weights_ss = torch.ones(config["nb_classes_inference"])
    list_ignore_classes = ignore_selection(config["ignore_idx"])
    for idx_ignore_class in list_ignore_classes: 
        weights_ss[idx_ignore_class] = 0
    logging.info(f"Ignored classes {list_ignore_classes}")
    logging.info(f"Weights of the different classes {weights_ss}")
    weights_ss= weights_ss.to(device)
    ce_loss_layer = torch.nn.CrossEntropyLoss(weight = weights_ss)
    
    ###For all classes, not just the inference one
    weights_ss_all = torch.ones(config["nb_classes"])
    list_ignore_classes = ignore_selection(config["ignore_idx"])
    for idx_ignore_class in list_ignore_classes: 
        weights_ss_all[idx_ignore_class] = 0
    logging.info(f"Ignored classes {list_ignore_classes}")
    logging.info(f"Weights of the different classes for all classes {weights_ss_all}")
    weights_ss_all= weights_ss_all.to(device)
    ce_loss_layer_all = torch.nn.CrossEntropyLoss(weight = weights_ss_all)
    

    net_final.eval()
    net_final.to(device)

    dict_with_mapping={}
    alpha=None

    list_parameter_to_update = []
    list_parameter_others = [] #2nd section of selected parameters, e.g. if scaling LL and Backbone differently

    net_final, list_parameter_to_update, list_parameter_others = \
        configure_freeze_models(net_final, config, list_parameter_to_update, list_parameter_others)


    for l_name, l_module in net_final.named_modules(): 
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()
            
    


    class_prior = np.zeros((1))
    class_prior, names_list = class_prior_class_names(config, logging)

    #Obtain the class priors from target training set
    alpha = 0
    alpha_tensor = (torch.ones(1)*alpha).to(device)
    
    #Weight for entropy loss
    ent_weigth = np.array(config["parameter"]["ent_weigth"]).astype(np.float64)
    ent_weigth = torch.from_numpy(ent_weigth).type(torch.FloatTensor).to(device)
    
    pl_weigth = np.array(config["parameter"]["pl_weigth"]).astype(np.float64)
    pl_weigth = torch.from_numpy(pl_weigth).type(torch.FloatTensor).to(device)
    
    summation_matrix = summation_matrix.to(device)
    class_prior = torch.from_numpy(class_prior).to(device)
    
    ### From https://github.com/DZhaoXd/DT-ST/blob/main/train_TCR_DTU.py#L325
    ###### confident  init 
    #default param in SAC (https://github.com/visinf/da-sac)
    THRESHOLD_BETA = 0.001
    running_conf = torch.zeros(config["nb_classes"]).cuda()
    running_conf.fill_(THRESHOLD_BETA)
    
    ###### Dynamic teacher init
    stu_eval_list = []
    stu_score_buffer = []
    res_dict = {'stu_ori':[], 'stu_now':[], 'update_iter':[]}

    #Initialisation of the Pseudo-Label backbone
    net_pseudo_label = make_a_deepcopy(net_final, logging)
    if config["parameter"]["ema_teacher"]:
        #If an ema techer is used
        net_his_optimizer = WeightEMA(
            list(net_pseudo_label.parameters()), 
            list(net_final.parameters()),
            alpha= 0.99)
        
    
    net_final.to(device)
    if config["parameter"]["finetune"] and config["parameter"]["fintune_setting"]=="classic":
        logging.info("Classifier get updated with 10X higher LR than backbone.")
        optimizer = torch.optim.AdamW([{"params": list_parameter_to_update, "lr":config["parameter"]["learning_rate"]},\
            {"params": list_parameter_others, "lr":config["parameter"]["learning_rate"] / 10.0}]) #Backbone is updated with a 10x smaller learning rate
    else: 
        optimizer = torch.optim.AdamW([{"params": list_parameter_to_update}],config["parameter"]["learning_rate"])
    if config["parameter"]["lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=config["parameter"]["nb_iterations"], power=0.9, last_epoch=-1, verbose=False)


    train_iter_trg = enumerate(target_train_loader)

    
    for i in range(config["parameter"]["nb_iterations"]):
        if i % config["parameter"]["val_intervall"]==0:
            logging.info(i)
            
            
            return_data_val_target_mapped = \
                validation_non_premap(net_final, config, target_test_loader, epoch=0, disable_log=False, device=device, list_ignore_classes=[0])
            logging.info(f"mIoU: {return_data_val_target_mapped['test_seg_head_miou']}")
            writer.add_scalar(f"validation.seg_mIou", return_data_val_target_mapped['test_seg_head_miou'], i)
            logging.info(f"Per class {return_data_val_target_mapped['seg_iou_per_class']}")
            for q in range(len(names_list)):
                writer.add_scalar(f"validation.seg_Iou_{names_list[q]}", return_data_val_target_mapped['seg_iou_per_class'][q], i) 
                
            
            #After validation, set again the BN to the defined setting
            for l_name, l_module in net_final.named_modules(): 
                if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
                    l_module.eval()
                    
            
            if i == 0:
                ###Initial pseudo labeling
                cls_thresh_cuda = pseudo_label(net_pseudo_label, config, target_train_loader, device, writer, names_list, i, running_conf, THRESHOLD_BETA)
                
        if i % config["parameter"]["ckpt_intervall"] == 0:
                torch.save({"state_dict": net_final.state_dict()},os.path.join(savedir_root, f"model_{i}.pth"),)    
                
        try:
            _, target_data = train_iter_trg.__next__()
        except:
            train_iter_trg = enumerate(target_train_loader)
            _, target_data = train_iter_trg.__next__()
            
            #New epoch so, recalculate the per class threshold with new model
            #Updating to latest model
            if config["parameter"]["ema_teacher"]:
                #No update if an EMA teacher is used
                pass
            else:
                net_pseudo_label = make_a_deepcopy(net_final, logging)
            
            #Creating the per class Thresholds
            cls_thresh_cuda = pseudo_label(net_pseudo_label, config, target_train_loader, device, writer, names_list, i, running_conf, THRESHOLD_BETA)


        new_data = copy.deepcopy(target_data)
        target_data = dict_to_device(target_data, device)
        optimizer.zero_grad()
        _, output_seg, _ = net_final.forward_mapped_learned(target_data)
        
        #### Entropy loss
        loss_ent = ent_weigth * minent_entropy_loss(output_seg)
        writer.add_scalar(f"training.entropy_loss",loss_ent, i)
        loss_seg = loss_ent
        
        
        
        ###Calculation of SND factor 
        output = F.softmax(output_seg.clone().detach(), dim=1).detach()[:,:,0]
        #With the SND criterion
        output_rand = torch.randperm(output.size(0))
        #select_point = pred1_rand.shape[0]
        select_point = 100
        pred1 = F.normalize(output[output_rand[:select_point]])
        pred1_en = entropy(torch.matmul(pred1, pred1.t()) * 20)
        writer.add_scalar(f"pseudo_label.training.SND",pred1_en, i)
        stu_score_buffer.append(pred1_en.item())
        stu_eval_list.append([new_data, output_rand.cpu()])
        
        thresolded_label=None
        with torch.no_grad():
            _, output_seg_pl, _ = net_pseudo_label.forward_mapped_learned_original(target_data)
            output_seg_pl = output_seg_pl.detach()
            
            ###DT-ST setting
            output_pl = F.softmax(output_seg_pl[:,:,0], dim=1)
            running_conf = update_running_conf(output_pl, running_conf, THRESHOLD_BETA)
            thresolded_label, _, _ = pseudo_labels_probs(output_pl, running_conf, THRESHOLD_BETA)
            thresolded_label = thresolded_label.detach()            
            
            mask_thresholded_label =  thresolded_label != 0 
            true_labels_pseudo_labels = target_data["y"].detach()[mask_thresholded_label]
            mask_ignore_points_in_pseudo_labels = true_labels_pseudo_labels==0
            pseudo_ignore_points_rate = np.sum(mask_ignore_points_in_pseudo_labels.detach().cpu().numpy()) / np.sum(mask_thresholded_label.cpu().numpy())
            writer.add_scalar(f"pseudo_label.training.pseudo_ignore_points_rate",pseudo_ignore_points_rate, i)
            
        if config["parameter"]["pl_no_mapping"]:
            #Do not do a mapping for the pseudo-labeling
            loss_pl=ce_loss_layer_all(output_seg[:,:,0], thresolded_label)
        else: 
            #Pseudo Label loss
            output_seg_merged=output_seg[:,:,0]@summation_matrix
            loss_pl = ce_loss_layer(output_seg_merged, thresolded_label)
        
        loss_pl*=pl_weigth
        writer.add_scalar(f"training.loss_pl", loss_pl, i)
        
        loss_seg=loss_seg+loss_pl
        
        writer.add_scalar(f"training.seg_loss", loss_seg, i)
        loss_seg.backward()
        optimizer.step()
        
        if config["parameter"]["lr_scheduler"]:
            writer.add_scalar(f"training.lr",  optimizer.param_groups[0]["lr"], i)
            scheduler.step()

        del loss_seg
        

        if config["parameter"]["fixed_update_iteration"]:
            if i % config["parameter"]["ema_update_iteration"] == 0:
                net_his_optimizer.step()
                logging.info("Updating the EMA Teacher at iteration {}".format(i))
            ## reset
            stu_eval_list = []
            stu_score_buffer = []
            
        else:
            if len(stu_score_buffer) >= 9 and int(len(stu_score_buffer)-9) % 3 ==0:   
                all_score = evel_stu(config, net_final, stu_eval_list, device, dict_with_mapping, alpha)
                compare_res = np.array(all_score) - np.array(stu_score_buffer)
                if np.mean(compare_res > 0) > 0.5 or len(stu_score_buffer) > 30:
                    update_iter = len(stu_score_buffer)
                    net_his_optimizer.step()
                    logging.info("Updating the EMA Teacher at iteration {}, with updater iter {}".format(i, update_iter))
                    
                    writer.add_scalar(f"pseudo_label.update_iteration", update_iter, i)
                    writer.add_scalar(f"pseudo_label.stu_ori", np.array(stu_score_buffer).mean(), i)
                    writer.add_scalar(f"pseudo_label.stu_now",np.array(all_score).mean(), i)
                    
                    ## reset
                    stu_eval_list = []
                    stu_score_buffer = []
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    #General settings
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--setting', '-ds', type=str, required=True, default="NS2SK")
    parser.add_argument('--resume_path', '-p', type=str, default="cvpr24_results/REP0_ns_semantic_TorchSparseMinkUNet_InterpAllRadiusNoDirsNet_1.0_trainSplit")
    parser.add_argument('--save_ckpt', '-scpt', type=bool, default=True)
    parser.add_argument('--tensorboard_folder', '-tf', type=str, default="UR")
    parser.add_argument('--bn_layer', '-l', type=str, default="standard")
    

    #Learning parameter
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--nb_iterations', '-i', type=int, default=20010)
    parser.add_argument('--ckpt_intervall', type=int, default=1000)
    parser.add_argument('--val_intervall', type=int, default=1000)
    parser.add_argument('--ent_weigth', '-ew', type=float, default=1.0)
    
    
    parser.add_argument('--lr_scheduler', '-ls', type=bool, default=False)
    parser.add_argument('--adaptive_weighting', '-aw', type=bool, default=False)

    #Select what to finetune
    parser.add_argument('--finetune', '-f', type=bool, default=False)
    parser.add_argument('--fintune_setting', '-fs', type=str, choices=['LL', 'classic', 'll_and_scalable_finetune', 'shot_finetune', 'complete_finetune'],  default='LL') #
    
    parser.add_argument('--prior_target', '-ps', type=bool, default=False)
    parser.add_argument('--free_bn_layer', '-b', type=bool, default=False)
    


    ### Pseudo-Label parameter
    parser.add_argument('--init_tgt_portion', type=float, default=0.20)
    parser.add_argument('--tgt_port_step', type=float, default=0.05)
    parser.add_argument('--max_tgt_port', type=float, default=0.5)
    
    parser.add_argument('--fixed_threshold', type=bool, default=False)
    parser.add_argument('--pl_no_mapping', type=bool, default=False)  #Indicates if we should do the pseudo-labelling with the class mapping or without
    
    parser.add_argument('--pl_weigth', '-plw', type=float, default=1.0)
    parser.add_argument('--DEBUG_remove_ignore_points_from_pl', type=bool, default=False)
    
    #EMA Teacher parameter
    parser.add_argument('--ema_teacher', type=bool, default=True)
    parser.add_argument('--ema_alpha', type=float, default=0.99)
    parser.add_argument('--ema_update_iteration', type=int, default=6)
    parser.add_argument('--fixed_update_iteration', type=bool, default=False)

    
    
    opts = parser.parse_args()

    config_arguments = {}
    config_arguments["name"] = opts.name
    config_arguments["resume_path"]=opts.resume_path
    config_arguments["tensorboard_folder"]=opts.tensorboard_folder
    config_arguments["setting"] = opts.setting
    config_arguments["bn_layer"] = opts.bn_layer
    
    config_arguments["finetune"] = opts.finetune
    config_arguments["fintune_setting"] = opts.fintune_setting
    config_arguments["free_bn_layer"] = opts.free_bn_layer
    config_arguments["ent_weigth"] = opts.ent_weigth
    config_arguments["prior_target"] = opts.prior_target

    config_arguments["save_ckpt"] = opts.save_ckpt
    config_arguments["learning_rate"] = opts.learning_rate
    config_arguments["batch_size"] = opts.batch_size
    config_arguments["nb_iterations"] = opts.nb_iterations
    config_arguments["ckpt_intervall"] = opts.ckpt_intervall
    config_arguments["val_intervall"] = opts.val_intervall 

    config_arguments["lr_scheduler"] = opts.lr_scheduler
    config_arguments["fixed_threshold"] = opts.fixed_threshold
    config_arguments["ema_teacher"]=opts.ema_teacher
    config_arguments["ema_alpha"]=opts.ema_alpha
    config_arguments["ema_update_iteration"]=opts.ema_update_iteration
    config_arguments["fixed_update_iteration"] = opts.fixed_update_iteration
    
    config_arguments["pl_weigth"] = opts.pl_weigth

    
    #Do the pseudo-labelling without class mapping 
    config_arguments["pl_no_mapping"] = opts.pl_no_mapping
    

    main(config_arguments)