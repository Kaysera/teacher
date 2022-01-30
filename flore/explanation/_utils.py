from functools import reduce


def alpha_factual_sum(explanations, alpha, debug=False):
    # This is the cummulative mu of the
    # rules that will be selected
    first_class_dict, first_matching, first_rule = explanations[0]
    total_mu = first_matching
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        total_mu += matching
        if debug:
            alpha_factual += [(rule, matching)]
        else:
            alpha_factual += [rule]

        if total_mu >= alpha:
            break

    if debug:
        return alpha_factual, total_mu
    else:
        return alpha_factual


def alpha_factual_diff(explanations, alpha, debug=False):
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [(first_rule, first_matching)]
    prev_matching = first_matching
    for class_dict, matching, rule in explanations[1:]:
        diff = prev_matching - matching
        if diff >= alpha:
            if debug:
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]

        prev_matching = matching
    if debug:
        total_mu = 0
        for rule, matching in alpha_factual:
            total_mu += matching
        return alpha_factual, total_mu
    else:
        return alpha_factual


def alpha_factual_factor(explanations, alpha, debug=False):
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [(first_rule, first_matching)]
    prev_matching = first_matching
    for class_dict, matching, rule in explanations[1:]:
        factor = prev_matching / matching
        if factor <= 1 + alpha:
            prev_matching = matching
            if debug:
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]
        else:
            break

    if debug:
        total_mu = 0
        for rule, matching in alpha_factual:
            total_mu += matching
        return alpha_factual, total_mu
    else:
        return alpha_factual


def alpha_factual_factor_sum(explanations, alpha, beta, debug=False):
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [(first_rule, first_matching)]
    prev_matching = first_matching
    total_mu = first_matching
    for class_dict, matching, rule in explanations[1:]:
        factor = prev_matching / matching
        if total_mu < beta or factor <= 1 + alpha:
            prev_matching = matching
            total_mu += matching
            if debug:
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]
        else:
            break

    if debug:
        return alpha_factual, total_mu
    else:
        return alpha_factual


def alpha_factual_avg(explanations, alpha, debug=False):
    avg = reduce(lambda x, y: x + y[1], explanations, 0) / len(explanations)
    first_class_dict, first_matching, first_rule = explanations[0]
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        if matching >= avg:
            if debug:
                alpha_factual += [(rule, matching)]
            else:
                alpha_factual += [rule]
        else:
            break

    if debug:
        total_mu = 0
        for rule, matching in alpha_factual:
            total_mu += matching
        return alpha_factual, total_mu
    else:
        return alpha_factual


def alpha_factual_robust(explanations, threshold, debug=False):
    # This is the cummulative mu of the
    # rules that will be selected
    first_class_dict, first_matching, first_rule = explanations[0]
    total_mu = first_matching
    alpha_factual = [(first_rule, first_matching)]
    for class_dict, matching, rule in explanations[1:]:
        if total_mu >= threshold:
            break
        total_mu += matching
        if debug:
            alpha_factual += [(rule, matching)]
        else:
            alpha_factual += [rule]



    if debug:
        return alpha_factual, total_mu
    else:
        return alpha_factual

