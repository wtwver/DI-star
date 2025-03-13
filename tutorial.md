=SLLearner._train(data)
logits, infer_action_info, hidden_state =  self._model.sl_train(**data, hidden_state)

==Model.sl_train()
lstm_input, = self.encoder(spatial_info, entity_info, scalar_info, entity_num)
lstm_output, out_state = self.core_lstm(lstm_input, hidden_state)
action_info, selected_units_num, logit = self.policy.train_forward(lstm_output,
return logits, action_info, out_state

===Model.policy.train_forward(lstm_output, scalar_context, action_info)
logit['action_type'], action['action_type'], embeddings = self.action_type_head(lstm_output, scalar_context, action_info['action_type'])
return action, selected_units_num, logit

====Model.policy.ActionTypeHead.forward(lstm_output, scalar_context, action_type)
return x, action_type, embedding

=SLLearner._train(data)
log_vars = self._loss.compute_loss(logits, data['action_info'], data['action_mask'],
                                               data['selected_units_num'], data['entity_num'], infer_action_info)

==SupervisedLoss.compute_loss(policy_logits, action, action_mask)
loss_dict = self._action_type_loss(policy_logits[loss_item_name], actions[loss_item_name], actions_mask[loss_item_name])
return loss_dict

===SupervisedLoss._action_type_loss(logits, label, mask)
loss_tmp = torch.nn.CrossEntropyLoss(logits, label)
loss_tmp *= mask
loss = loss_tmp.sum() / valid_num
return loss

=SLLearner._train(data)
loss = log_vars['total_loss']
if self.ignore_step > 5: # warm up
	self._optimizer.zero_grad()
	loss.backward()