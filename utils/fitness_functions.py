


def double_pole_fitness_func(target_len, cart, net):

    def fitness_func(genes):
        net.init_weight(genes)
        return net.evaluate(cart, target_len)
        

    return fitness_func



