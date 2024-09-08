from utils.utils import (get_bbdir_root, get_savedir_root)
from utils_source_free.general_imports import * 
from utils_source_free.utils import (summation_matrix_generator, sf_class_mapping_loader)




def validation_consistency(net, net_later, config, test_loader, epoch, disable_log, device, criterion_model_train=True):
    
    net.eval()
    if criterion_model_train: 
        print("Criterion model is set to train mode")
        net_later.train()
    else:
        print("Criterion model is set to Eval mode")
        net_later.eval()
    cons_rate_sum = 0.0

    
    cons_number = 0
    cm_seg_head = np.zeros((config["nb_classes_inference"],config["nb_classes_inference"]))
    cm_seg_head_later = np.zeros((config["nb_classes_inference"],config["nb_classes_inference"]))

    mapping_information = sf_class_mapping_loader(source_dataset=config["source_dataset_name"], target_dataset=config["target_dataset_name"])
    
    with torch.no_grad():
        count_iter = 0
        
        for data in test_loader:
            
            data = dict_to_device(data, device)
            
            label = data["y"].cpu().detach().numpy()
            
            
            _, output_seg, _ = net.forward_pretraining_inside(data) #SOURCE data
            if "no_premap" in config and config["no_premap"]:
                output_seg = output_seg[:,:,0]@config["summation_matrix"]
            else:
                output_for_validation =  prediction_changer(output_seg.cpu().detach(), mapping_information)
            
            output_seg_np = np.argmax(output_seg[:,1:].cpu().detach().numpy(), axis=1) + 1

            
            cm_seg_head_tmp = confusion_matrix(label.ravel(), output_for_validation.ravel(), labels=list(range(config["nb_classes_inference"])))
            cm_seg_head += cm_seg_head_tmp
            
           
            _, output_seg_later, _ = net_later.forward_pretraining_inside(data) #SOURCE data
            if "no_premap" in config and config["no_premap"]:
                output_seg_later = output_seg_later[:,:,0]@config["summation_matrix"]
            else:
                output_for_validation_later =  prediction_changer(output_seg_later.cpu().detach(), mapping_information)
                
            output_seg_np_later = np.argmax(output_seg_later[:,1:].cpu().detach().numpy(), axis=1) + 1   
            
            cm_seg_head_tmp_later = confusion_matrix(label.ravel(), output_for_validation_later.ravel(), labels=list(range(config["nb_classes_inference"])))
            cm_seg_head_later += cm_seg_head_tmp_later
            
            #Absolute Consistence
            absolute_consistence = np.sum(np.equal(output_seg_np, output_seg_np_later))
            consistence_rate = absolute_consistence / output_seg_np.shape[0]
            cons_rate_sum = cons_rate_sum + consistence_rate
            
           
            
            cons_number = cons_number + 1
            count_iter += 1
            if count_iter % 10 == 0:
                torch.cuda.empty_cache()

    consistency_rate = cons_rate_sum / cons_number
    
    
    test_seg_head_miou, _ = metrics.stats_iou_per_class(cm_seg_head, ignore_list=[0]) #First return value is the mean IoU
    test_seg_head_miou_later, _ = metrics.stats_iou_per_class(cm_seg_head_later, ignore_list=[0]) #First return value is the mean IoU
    
    
    return consistency_rate, test_seg_head_miou, test_seg_head_miou_later

def main(config_arguments):
    # define the logging
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter('runs_eccv/{}'.format(f"{config_arguments['tensorboard_folder']}/{config_arguments['name']}"))
    
    file_path_config = os.path.join(config_arguments["resume_path"], "config.yaml")
    config = read_yaml_file(file_path_config)

    config["da_fixed_head_path_model"]= config_arguments["resume_path"]
    config["parameter"] = {}
    config["parameter"] = config_arguments
    config["network_backbone"] = 'TorchSparseMinkUNet_learned'
    config["test_batch_size"] =  config_arguments["bs"]
    config["training_batch_size"] =  config_arguments["bs"]
    config["no_premap"] = config_arguments["no_premap"]
     
    experiment_name = "Debug"
    # device
    device = torch.device(config['device'])
    if config["device"] == "cuda":
        torch.backends.cudnn.benchmark = True


    # define the logging
    logging.getLogger().setLevel(config["logging"])

    # get the savedir
    savedir_root = get_savedir_root(config, experiment_name)
    bb_dir_root = get_bbdir_root(config)
    # create the network
    disable_log = not config["interactive_log"]
    latent_size = config["network_latent_size"]
    backbone = config["network_backbone"]

    in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
    
    logging.info("Creating the network")
    def network_function():
        return networks.Network(in_channels=in_channels_source, latent_size=latent_size, backbone=backbone,\
                    voxel_size=config["voxel_size"], dual_seg_head = config["dual_seg_head"], target_in_channels=in_channels_target, config=config)
    net = network_function()
    
    
    net.to(device)
    
    logging.info(f"Network -- Number of parameters {count_parameters(net)}")
    logging.info("Getting the dataset")
    target_DatasetClass = get_dataset(eval("datasets."+config["target_dataset_name"]))
    
    val_number = 1 #1: verifying split, 2 train split, else: test split
    
    dataloader_dict = da_sf_get_dataloader(target_DatasetClass, config, net, network_function, val=val_number)
    
    target_train_loader = dataloader_dict ["target_train_loader"]
    target_test_loader = dataloader_dict ["target_test_loader"]

    list_pths = [(x, x+config["parameter"]["cons_intervall"]) for x in range(config["parameter"]["nb_iterations_start"], config["parameter"]["nb_iterations"], config["parameter"]["cons_intervall"]) if x+config["parameter"]["cons_intervall"] <=config["parameter"]["nb_iterations"]]
    print(f"Available list pths {list_pths}")
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(config)), os.path.join(savedir_root, "config.yaml"))
    epoch_start = 0
    
    net_target = network_function()
    
    
    net_target.to(device)
    
    
        
    for paths_compare in list_pths:
        print(paths_compare)
        ## ckpt_number = -1 means to load the last ckpt
        ckpt_path = os.path.join(bb_dir_root, f'model_{paths_compare[0]}.pth')
        logging.info(f"CKPT -- Load ckpt from {ckpt_path}")
        #Load the checkpoint for the backbone
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
        except: 
            print(f"not available: {ckpt_path}")
        #Load the pretrained model

        try: 
            net.load_state_dict(checkpoint["state_dict"])
        except Exception as e: 
            print(e)
            logging.info(f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false")
            net.load_state_dict(checkpoint["state_dict"], strict=False)

        if config["parameter"]["a_c_ckpt_flag"]:
            #If we want to compare to a specific model
             ckpt_path = os.path.join(config["parameter"]["a_c_ckpt_path"])
        else: 
            if config["parameter"]["compare_first_model"]:
                #Compare to the source only model
                ckpt_path = os.path.join(bb_dir_root, f'model_{list_pths[0][0]}.pth')
            else: 
                ckpt_path = os.path.join(bb_dir_root, f'model_{paths_compare[0]}.pth')

        logging.info(f"CKPT -- Load Criterion ckpt from {ckpt_path}")

        #Load the checkpoint for the backbone
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
        except: 
            print(f"not available: {ckpt_path}")
        #Load the pretrained model

        try: 
            net_target.load_state_dict(checkpoint["state_dict"])
        except Exception as e:
            print(e)
            logging.info(f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false")
            net_target.load_state_dict(checkpoint["state_dict"], strict=False)

        
        
        epoch_start = 0
        
        # create the loss layer
        weights_ss = torch.ones(config["nb_classes"])
        list_ignore_classes = ignore_selection(config["ignore_idx"])
        for idx_ignore_class in list_ignore_classes: 
            weights_ss[idx_ignore_class] = 0
        logging.info(f"Ignored classes {list_ignore_classes}")
        logging.info(f"Weights of the different classes {weights_ss}")
        weights_ss= weights_ss.to(device)
        epoch = epoch_start
        
        mapping_info = sf_class_mapping_loader(source_dataset=config["source_dataset_name"], target_dataset=config["target_dataset_name"])
        summation_matrix = summation_matrix_generator(mapping_info)
        summation_matrix = summation_matrix.to(device)
        config["summation_matrix"]=summation_matrix
        
        #Should we put the criterion model in train mode (instance norm)
        train_mode = not config['parameter']['not_train_mode']
        
        if False :
            consistency_rate_train, seg_miou_train, seg_miou_train_later = validation_consistency(net, net_target, config, target_train_loader, epoch, disable_log, device, criterion_model_train=train_mode)
            writer.add_scalar(f"evaluator.train.consistency_rate", consistency_rate_train, int(paths_compare[0]))
            writer.add_scalar(f"evaluator.train.seg_miou", seg_miou_train, int(paths_compare[0]))
            writer.add_scalar(f"evaluator.train.seg_miou_later", seg_miou_train_later, int(paths_compare[0]))
        
        
        consistency_rate_test, seg_miou_test, seg_miou_test_later = validation_consistency(net, net_target, config, target_test_loader, epoch, disable_log, device, criterion_model_train=train_mode)
        writer.add_scalar(f"evaluator.test.consistency_rate", consistency_rate_test, int(paths_compare[0]))
        writer.add_scalar(f"evaluator.test.seg_miou", seg_miou_test, int(paths_compare[0]))
        writer.add_scalar(f"evaluator.test.seg_miou_later", seg_miou_test_later, int(paths_compare[0]))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--resume_path', '-p', type=str, default="tta_results/EVAL_AdaBN_REP_Cos_Baseline_ns_sk_final_TorchSparseMinkUNet_InterpAttentionKHeadsNet_32_trainSplit")
    parser.add_argument('--tensorboard_folder', '-tf', type=str, default="Evaluator")
    parser.add_argument('--no_premap', '-pm', type=bool, default=False)
    parser.add_argument('--nb_iterations_start', type=int, default=0)
    parser.add_argument('--nb_iterations', type=int, default=20001)
    parser.add_argument('--cons_intervall', type=int, default=1000)
    parser.add_argument('--compare_first_model', type=bool, default=False)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--bn_layer', '-l', type=str, default="standard")
    parser.add_argument('--a_c_ckpt_flag', type=bool,  default=False) #When we want to compare explicitely to another ckpt, set this flag to true
    parser.add_argument('--a_c_ckpt_path', type=str,  default='NONE') #When we want to compare explicitely to another ckpt, give here the path
    parser.add_argument('--not_train_mode', type=bool,  default=False) #Set to True when you want to use eval mode of criterion
    parser.add_argument('--tent_flag', type=bool, default=False)
    parser.add_argument('--validator_bn_layer', type=str, default="None")
    
    
    opts = parser.parse_args()

    config_arguments = {}
    config_arguments["name"] = opts.name
    config_arguments["resume_path"] = opts.resume_path
    config_arguments["tensorboard_folder"]=opts.tensorboard_folder
    config_arguments["no_premap"]=opts.no_premap
    config_arguments["nb_iterations"] = opts.nb_iterations
    config_arguments["cons_intervall"] = opts.cons_intervall
    config_arguments["compare_first_model"] = opts.compare_first_model
    config_arguments["nb_iterations_start"] = opts.nb_iterations_start
    config_arguments["bs"] = opts.bs
    config_arguments["bn_layer"] = opts.bn_layer
    
    config_arguments["a_c_ckpt_flag"] = opts.a_c_ckpt_flag
    config_arguments["a_c_ckpt_path"] = opts.a_c_ckpt_path
    config_arguments["not_train_mode"] = opts.not_train_mode
    config_arguments["tent_flag"] = opts.tent_flag
    config_arguments["validator_bn_layer"] = opts.validator_bn_layer



    main(config_arguments)