from utils_source_free.general_imports import * 


def main(config_arguments):
    #Setting the seeds
    torch.manual_seed(1234) 
    random.seed(1234)
    np.random.seed(1234)
    
    # define the logging
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter('runs_eccv/{}'.format(f"{config_arguments['tensorboard_folder']}/{config_arguments['name']}"))

    file_path_config = os.path.join(config_arguments["resume_path"], "config.yaml")
    config = read_yaml_file(file_path_config)


    if config is not None:
    # Access the data in the dictionary
        print(file_path_config)
        print(f"Loaded Config: {config}")    
    else:
        print("Failed to read the config YAML file.")
        return 0
    
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

    in_channels_source, _, in_channels_target, _ = da_get_inputs(config)
    

    logging.info("Creating the network")
    def network_function():
        return networks.Network(in_channels=in_channels_source, latent_size=latent_size, backbone=backbone,\
                    voxel_size=config["voxel_size"], dual_seg_head = config["dual_seg_head"], target_in_channels=in_channels_target, config=config)

    ### Final network
    net_final = network_function()
    
    ckpt_path = os.path.join(bb_dir_root, 'source_only.pth')
    logging.info(f"CKPT -- Load ckpt from {ckpt_path}")

    #Load the checkpoint for the backbone
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
        logging.info(f"Loaded parameters do not match exactly net architecture, switching to load_state_dict strict=false.")
        net_final.load_state_dict(checkpoint_new, strict=False)

    logging.info(f"Network -- Number of parameters {count_parameters(net_final)}")
 
    target_DatasetClass = get_dataset(eval("datasets."+config["target_dataset_name"]))

    val_number = 1 #1: verifying split, 2 train split, else: test split

    
    dataloader_dict = da_sf_get_dataloader(target_DatasetClass, config, net_final, network_function, val=val_number, train_shuffle=True, keep_orignal_data=False)

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

    net_final.eval()
    net_final.to(device)

    list_parameter_to_update = []
    list_parameter_others = [] #2nd section of selected parameters, e.g. if scaling LL and Backbone differently

    net_final, list_parameter_to_update, list_parameter_others = \
        configure_freeze_models(net_final, config, list_parameter_to_update, list_parameter_others)


    for l_name, l_module in net_final.named_modules(): 
        if isinstance(l_module, torch.nn.modules.batchnorm._BatchNorm):
            l_module.eval()
           


    class_prior = np.zeros((1))
    class_prior, names_list = class_prior_class_names(config, logging)
    
    
    #Renormalize the class_prior to have numerical exactly a sum of 1 
    class_prior = class_prior / np.sum(class_prior) 
    logging.info(f"We use a distribution of {class_prior}")
    
    ent_loss_thr = np.array(config["parameter"]["ent_loss_thr"]).astype(np.float64)
    ent_loss_thr = torch.from_numpy(ent_loss_thr).type(torch.FloatTensor).to(device)

    div_loss_thr = np.array(config["parameter"]["div_loss_thr"]).astype(np.float64)
    div_loss_thr = torch.from_numpy(div_loss_thr).type(torch.FloatTensor).to(device)

    summation_matrix = summation_matrix.to(device)

    class_prior = torch.from_numpy(class_prior).to(device)


    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    net_final.to(device)
    
    if config["parameter"]["finetune"] and config["parameter"]["fintune_setting"]=="classic":
        logging.info("Classifier get updated with 10X higher LR than backbone.")
        optimizer = torch.optim.AdamW([{"params": list_parameter_to_update, "lr":config["parameter"]["learning_rate"]},\
            {"params": list_parameter_others, "lr":config["parameter"]["learning_rate"] / 10.0}]) #Backbone is updated with a 10x smaller learning rate
    else: 
        optimizer = torch.optim.AdamW([{"params": list_parameter_to_update}],config["parameter"]["learning_rate"])
    
    
    
    if config["parameter"]["lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20005, eta_min=0)

    logging.info(f"Network -- Number of finally optimized parameters {count_parameters(net_final)}")
    train_iter_trg = enumerate(target_train_loader)

    for i in range(config["parameter"]["nb_iterations"]):
        if i % config["parameter"]["val_intervall"]==0:
            logging.info(i)
            
            
            #if "code_test" in config and config["code_test"]:
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
                    
            
        if i % config["parameter"]["ckpt_intervall"]==0:
                torch.save({"state_dict": net_final.state_dict()},os.path.join(savedir_root, f"model_{i}.pth"),)
            
        try:
            _, target_data = train_iter_trg.__next__()
        except:
            train_iter_trg = enumerate(target_train_loader)
            _, target_data = train_iter_trg.__next__()


        target_data = dict_to_device(target_data, device)
        optimizer.zero_grad()
        _, output_seg, _ = net_final.forward_mapped_learned(target_data)

        loss_seg = None
        #### Entropy loss
        loss_ent = minent_entropy_loss(output_seg)
        loss_ent = F.relu(loss_ent - ent_loss_thr, inplace=False)
        loss_seg = loss_ent
        writer.add_scalar(f"training.entropy_loss",loss_seg, i)

        #### Diversity loss
        nb_points = output_seg.shape[0]
        #Mapping to the new class output
        output_seg = output_seg[:,:,0]@summation_matrix
        input = F.softmax(output_seg[:,1:], dim=1).sum(dim=0)/nb_points
        input_log = torch.log(input)
        loss_kl = kl_loss(input_log, class_prior).type(torch.FloatTensor)
        loss_kl =  F.relu(loss_kl - div_loss_thr, inplace=False)
        div_loss =  loss_kl
        writer.add_scalar(f"training.diversity_loss", div_loss, i)
        loss_seg = loss_seg+div_loss


        writer.add_scalar(f"training.seg_loss", loss_seg, i)
        loss_seg.backward()
        optimizer.step()
        
        del loss_seg


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    #General settings
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--setting', '-ds', type=str, required=True, default="NS2SK")
    parser.add_argument('--resume_path', '-p', type=str, default="cvpr24_results/REP0_ns_semantic_TorchSparseMinkUNet_InterpAllRadiusNoDirsNet_1.0_trainSplit")
    parser.add_argument('--tensorboard_folder', '-tf', type=str, default="DASF")
    parser.add_argument('--bn_layer', '-l', type=str, default="standard")

    #Learning parameter
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)
    parser.add_argument('--batch_size', '-bs', type=int, default=4)
    parser.add_argument('--nb_iterations', '-i', type=int, default=20010)
    parser.add_argument('--ckpt_intervall', type=int, default=1000)
    parser.add_argument('--val_intervall', type=int, default=1000)
    parser.add_argument('--lr_scheduler', '-ls', type=bool, default=False)
    

    #Select what to finetune
    parser.add_argument('--finetune', '-f', type=bool, default=False)
    parser.add_argument('--fintune_setting', '-fs', type=str, choices=['LL', 'classic', 'll_and_scalable_finetune', 'shot_finetune', 'complete_finetune'],  default='LL') #

    #Clipping the loss
    parser.add_argument('--ent_loss_thr', '-eth', type=float, default=0.04)
    parser.add_argument('--div_loss_thr', '-dth', type=float, default=0.04)



    opts = parser.parse_args()

    config_arguments = {}
    #Experiment credentials
    config_arguments["name"] = opts.name
    config_arguments["tensorboard_folder"] = opts.tensorboard_folder
    config_arguments["resume_path"] = opts.resume_path

    #Training settings
    config_arguments["setting"] = opts.setting
    config_arguments["bn_layer"] = opts.bn_layer
    config_arguments["finetune"] = opts.finetune
    config_arguments["fintune_setting"] = opts.fintune_setting

    
    config_arguments["learning_rate"] = opts.learning_rate
    config_arguments["batch_size"]= opts.batch_size
    config_arguments["nb_iterations"] = opts.nb_iterations
    
    #Evaluation settings
    config_arguments["ckpt_intervall"] = opts.ckpt_intervall
    config_arguments["val_intervall"] = opts.val_intervall 

    #Clipping
    config_arguments["ent_loss_thr"] = opts.ent_loss_thr
    config_arguments["div_loss_thr"] = opts.div_loss_thr
    config_arguments["lr_scheduler"] = opts.lr_scheduler

  
    main(config_arguments)