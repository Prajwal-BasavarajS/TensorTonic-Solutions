def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = []

    for s in range(num_states):
        best = float('-inf')  # initialize to negative infinity

        for a in range(len(transitions[s])):
            q = rewards[s][a]

            # expected future value
            for s_next in range(num_states):
                q += gamma * transitions[s][a][s_next] * values[s_next]

            best = max(best, q)

        new_values.append(float(best))

    return new_values