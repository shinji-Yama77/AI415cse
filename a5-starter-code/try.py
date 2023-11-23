#valid_operators = [o for o in all_opers if o.pre_condition]
    #all_opers = tm.Operator.all_operators()
    #for operator in valid_operators:
            #val = 0
            #action = operator.name
            #possible_states = [possible for possible in operator.state_transfer(state)]
            # only one state is possible from applying the operator, so we want the sum of 
            #for op, next_state in mdp.state_graph[state]:

            #for possible in possible_states:
                #val += mdp.transition(state, action, possible) * (mdp.reward(state, action, possible) + v_table[possible])
            #q_table[(state, action)] = val
            #current_max_val = max(current_max_val, val)
        #new_v_table[state] = current_max_val
        #max_delta = max(max_delta, abs(v_table[state]-current_max_val))


        #for operator, next_state in mdp.state_graph[state]: # not actually all the valid actions
            #curr_reward = mdp.reward(state, operator.name, next_state)
            #val = 0
            #for op, ns in mdp.state_graph[state]:
                #gamma = mdp.config.gamma
                #val += mdp.transition(state, op.name, ns) + (gamma * v_table[ns])
                #val += mdp.transition(state, operator.name, ns) * (mdp.reward(state, operator.name, ns) + (gamma * v_table[ns]))
            ##q_table[(state, operator.name)] = val
            #current_max_val = max(current_max_val, val)
        #new_v_table[state] = current_max_val
        #max_delta = max(max_delta, abs(v_table[state]-new_v_table[state]))