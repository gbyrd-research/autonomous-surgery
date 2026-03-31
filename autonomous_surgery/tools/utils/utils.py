def forward_pass(data, policy):
    
    images, state, action_chunk, action_is_pad, instruction_text = data
    
    images_data, qpos_data, action_data, is_pad = (
        {img_name: img.cuda() for img_name, img in images.items()},
        state.cuda(),
        action_chunk.cuda(),
        action_is_pad.cuda(),
    )
    # TODO: We may need to create command embeddings for the instruction text.
    # For now, we will not use this
    return policy(qpos_data, images_data, action_data, is_pad)

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d