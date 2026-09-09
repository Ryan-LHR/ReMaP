"""
The main workflow
"""

from parse import *
from utils import *


def main(args):
    """ ==============================
              (1) Loading
    ============================== """
    save_point_print('(1) Loading...')
    device = get_device(args.device)

    model = load_model(args.model, args.model_class, args.model_file, device)
    print(f"\nThe total number of Parameters in the model: {get_model_parameters(model)}")

    train_loader, test_loader = load_data(dataset_name=args.dataset, path_to_data=args.dataset_dir,
                                          batch_size=args.batch_size, n_workers=args.n_workers, args=args)

    """ ==============================
      (2) Candidate Set Constructing
    ============================== """
    save_point_print('(2) Candidate Set Constructing...')
    kwargs = {'model': model, 'model_name': args.model, 'test_loader': test_loader}
    cand_set, cand_loader = construct_candidate(dataset_name=args.dataset, path_to_data=args.dataset_dir,
                                                test_set=test_loader.dataset, cand_type=args.cand_type,
                                                cand_size=args.cand_size, batch_size=args.batch_size,
                                                include_test=True, args=args, **kwargs)

    if args.debug_mode:
        train_loader, test_loader, cand_set, cand_loader = \
            downsample_debug(train_loader, test_loader, cand_set, cand_loader, args)

    cand_truths = extract_truths(cand_loader.dataset)

    """ ==============================
             (3) Predicting
    ============================== """
    save_point_print('(3) Predicting...')
    if not args.load_prediction:
        (train_vectors, train_labels, train_correct) = get_predictions(
            model, train_loader, return_type='Tensor', is_print=True, data_name='Train', device=device)
        (test_vectors, test_labels, test_correct) = get_predictions(
            model, test_loader, return_type='Tensor', is_print=True, data_name='Test', device=device)
        (cand_vectors, cand_labels, cand_correct) = get_predictions(
            model, cand_loader, return_type='Tensor', is_print=True, data_name='Candidate', device=device)

    else:
        base_kwargs = {'model': model, 'device': device}

        train_kwargs = {**base_kwargs, 'loader': train_loader}
        test_kwargs = {**base_kwargs, 'loader': test_loader}
        cand_kwargs = {**base_kwargs, 'loader': cand_loader}

        (train_vectors, train_labels, train_correct) = load_prediction_results(args, label='Train', **train_kwargs)
        (test_vectors, test_labels, test_correct) = load_prediction_results(args, label='Test', **test_kwargs)
        (cand_vectors, cand_labels, cand_correct) = load_prediction_results(args, label=f'Candidate_{args.cand_type}', **cand_kwargs)

    if args.cand_type == 'nominal':
        check_consistency(test_labels, cand_labels)

    """ ==============================
      (4) Prioritization / Selection
    ============================== """
    save_point_print('(4) Prioritizing...')
    start_time = time.time()
    prioritized_result = dispatch_method(
        model=model,
        train_loader=train_loader,
        train_vectors=train_vectors,
        train_labels=train_labels,
        train_correct=train_correct,
        cand_set=cand_set,
        cand_loader=cand_loader,
        cand_truths=cand_truths,
        cand_vectors=cand_vectors,
        cand_labels=cand_labels,
        cand_correct=cand_correct,
        device=device,
        args=args,
    )

    end_time = time.time()
    exec_time_dict = measure_execution_time(start_time, end_time, stage='Execution Time')

    """ ==============================
             (5) Evaluation
    ============================== """
    save_point_print('(5) Evaluating...')

    if args.method not in SELECTION_METHOD:
        prioritized_indices = prioritized_result
        # Check the uniqueness of elements in the prioritized list
        if len(prioritized_indices) != len(set(prioritized_indices)):
            raise ValueError("prioritized indices contains duplicate elements!!!")

        # Evaluate overall metrics
        pri_correct = [cand_correct[i] for i in prioritized_indices]
        result_dict = eval_overall(pri_correct, args.budget_list)

        # Evaluate fault types
        pri_pred_labels = [cand_labels[i].item() for i in prioritized_indices]
        pri_true_labels = [cand_truths[i] for i in prioritized_indices]
        fault_types_dict = eval_fault_types(pri_pred_labels, pri_true_labels, args.budget_list)

    elif args.method in SELECTION_METHOD:
        prioritized_dict = prioritized_result
        result_dict = eval_overall_selection(prioritized_dict, cand_correct)
        fault_types_dict = eval_fault_types_selection(prioritized_dict, cand_labels, cand_truths)

    result_dict.update(fault_types_dict)
    result_dict.update(exec_time_dict)
    result_dict.update({"Start Timestamp": args.ex_time})
    result_dict.update({"Machine Name": args.machine_name})

    # Save prioritized indices
    if args.save_result:
        save_path_indices = get_indices_save_path(args)
        save_to_pickle(save_path_indices, prioritized_result)

    return result_dict


def parse_args():
    # Initialize arguments
    parser = argparse.ArgumentParser(description='Initialize arguments')
    parser_add_args(parser)
    args = parser.parse_args()
    args = modify_args(args)

    # Prepare
    save_path_csv = get_result_save_path(args)
    result_list = []

    # Run experiments
    for args.seed in args.seed_list:
        # Print experiment setup
        settings = get_settings(args)
        print_msg_box(settings, title='Experiment Settings')

        # Run experiment
        result_dict = main(args)

        # Print and Save results
        result_dict = print_results(args.rq, args.method, result_dict, args.seed)
        save_result_to_csv(save_path_csv, result_dict, args)
        save_result_dict(result_dict, args, result_type='single')
        result_list.append(result_dict)

    # Calculate average result
    save_point_print('Overall Average Result')
    overall_result_dict = get_result_overall(result_list)

    # Save average result
    overall_result_dict = print_results(args.rq, args.method, overall_result_dict)
    save_result_to_csv(save_path_csv, overall_result_dict, args)
    save_result_dict(overall_result_dict, args, result_type='avg')

    return result_dict
