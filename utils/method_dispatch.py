from baselines import *

from .dufp.dufp import prioritize_by_dufp
from .remap.remap import prioritize_by_remap


COVERAGE_BASED = ['nac-ctm', 'nbc-ctm', 'kmnc-ctm', 'snac-ctm',
                  'nac-cam', 'nbc-cam', 'kmnc-cam', 'snac-cam', 'tknc-cam']
SURPRISE_BASED = ['lsa', 'pc-dsa', 'pc-lsa', 'pc-mlsa', 'pc-mdsa', 'pc-mmdsa']
CONFIDENCE_BASED = ['deepgini', 'maxp', 'margin', 'entropy']
SELECTION_METHOD = ['mcp_official', 'datis', 'sets',
                    "best_ratio(0.0)", "best_ratio(0.25)", "best_ratio(0.5)",
                    "best_ratio(0.75)", "best_ratio(1.0)"]

def dispatch_method(
        model,
        train_loader,
        train_vectors,
        train_labels,
        train_correct,
        cand_set,
        cand_loader,
        cand_truths,
        cand_vectors,
        cand_labels,
        cand_correct,
        device,
        args,
):
    """
    Dispatch Prioritization or Selection by method name
    """
    if args.method not in SELECTION_METHOD:
        """
        Prioritization Method: 
            Sort all samples globally once. 
            When k changes, simply extract the top-k% samples from the sorted sequence without re-ranking.
        """
        if 'dufp' in args.method:
            prioritized_indices = prioritize_by_dufp(model, args.model, train_loader, train_vectors, train_labels, cand_loader,
                                                      cand_truths, cand_vectors, cand_labels, cand_correct,
                                                      device, args)
        elif 'remap' in args.method:
            prioritized_indices = prioritize_by_remap(model, args.model, train_loader, train_vectors, train_labels, cand_loader,
                                                      cand_truths, cand_vectors, cand_labels, cand_correct,
                                                      device, args)
        elif args.method in COVERAGE_BASED:
            prioritized_indices = prioritize_by_coverage(args.method, model, train_loader,
                                                         cand_loader, args.seed)

        elif args.method in SURPRISE_BASED:
            prioritized_indices = prioritize_by_surprise(args.method, model, args.model,
                                                         train_loader, cand_loader, cand_labels,
                                                         args.seed, device, args)
        elif args.method == 'lof' or args.method.startswith('lof_'):
            prioritized_indices = prioritize_by_lof(model, args.model, train_loader,
                                                    cand_loader, device, args)
        elif args.method in CONFIDENCE_BASED:
            prioritized_indices = prioritize_by_confidence(args.method, cand_vectors, device)

        elif args.method == 'ats':
            prioritized_indices = prioritize_by_ats(train_loader, cand_loader, cand_vectors, cand_labels, args, args.model)

        elif args.method == 'nns':
            prioritized_indices = prioritize_by_nns(cand_vectors)

        elif args.method == 'nss':
            prioritized_indices = prioritize_by_nss(model, args.model, args.dataset, cand_loader, args.seed)

        elif args.method == 'rts':
            prioritized_indices = prioritize_by_rts(args.dataset, train_loader.dataset, cand_set,
                                                    train_vectors, train_labels, cand_vectors, args.seed)

        elif args.method == 'certpri':
            prioritized_indices = prioritize_by_certpri(model, cand_set, cand_vectors, args.seed, device)

        elif args.method == 'fast':
            prioritized_indices = prioritize_by_fast(model, args.model, train_loader, train_loader.dataset, train_vectors,
                                                     train_correct, cand_loader, cand_labels, args.batch_size, args)
        elif args.method == 'prima':
            prioritized_indices = prioritize_by_prima(model, args.model, train_loader.dataset, train_vectors,
                                                     train_correct, cand_loader, cand_labels, args.batch_size,
                                                      args.data_dir, args.cand_type, args.seed)
        elif args.method == 'random':
            prioritized_indices = prioritize_by_random(args.seed, cand_set)
        elif args.method == 'best':
            prioritized_indices = prioritize_by_best(cand_correct, args.seed)
        elif args.method == 'datis_no_selection':
            prioritized_indices = prioritize_by_datis(model, args.model, train_loader, cand_loader,
                                                   cand_vectors, cand_labels, args.budget_list, args, train_correct,
                                                           train_vectors)
        prioritized_result = prioritized_indices

    elif args.method in SELECTION_METHOD:
        """
        Selection Method: 
            Dynamically select the top-k% samples. 
            Each time k changes, re-selection is performed based on the updated criteria.
        """
        if args.method == 'mcp_official':
            prioritized_dict = prioritize_by_mcp_official(cand_vectors, args.budget_list)
        elif args.method == 'datis':
            prioritized_dict = prioritize_by_datis(model, args.model, train_loader, cand_loader,
                                                   cand_vectors, cand_labels, args.budget_list, args, train_correct,
                                                           train_vectors)
        elif args.method == 'sets':
            prioritized_dict = prioritize_by_sets(model, args.model, train_loader, cand_loader,
                                                  cand_vectors, args.budget_list, args)
        elif "best_ratio" in args.method:
            prioritized_dict = selection_by_best_ratio(cand_correct, args.budget_list, args.method, args.seed)
        prioritized_result = prioritized_dict
    else:
        raise ValueError("Method should be set!")


    return prioritized_result
