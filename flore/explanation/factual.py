def get_factual_FID3(instance, rule_list, threshold=0.01):
    return [rule for rule in rule_list if rule.matching(instance) > threshold]
