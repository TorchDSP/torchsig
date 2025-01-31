torchsig.models.iq\_models.inceptiontime.inceptiontime.InceptionTime
====================================================================

.. currentmodule:: torchsig.models.iq_models.inceptiontime.inceptiontime

.. autoclass:: InceptionTime
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~InceptionTime.add_module
      ~InceptionTime.all_gather
      ~InceptionTime.apply
      ~InceptionTime.backward
      ~InceptionTime.bfloat16
      ~InceptionTime.buffers
      ~InceptionTime.children
      ~InceptionTime.clip_gradients
      ~InceptionTime.compile
      ~InceptionTime.configure_callbacks
      ~InceptionTime.configure_gradient_clipping
      ~InceptionTime.configure_model
      ~InceptionTime.configure_optimizers
      ~InceptionTime.configure_sharded_model
      ~InceptionTime.cpu
      ~InceptionTime.cuda
      ~InceptionTime.double
      ~InceptionTime.eval
      ~InceptionTime.extra_repr
      ~InceptionTime.float
      ~InceptionTime.forward
      ~InceptionTime.freeze
      ~InceptionTime.get_buffer
      ~InceptionTime.get_extra_state
      ~InceptionTime.get_parameter
      ~InceptionTime.get_submodule
      ~InceptionTime.half
      ~InceptionTime.ipu
      ~InceptionTime.load_from_checkpoint
      ~InceptionTime.load_state_dict
      ~InceptionTime.log
      ~InceptionTime.log_dict
      ~InceptionTime.lr_scheduler_step
      ~InceptionTime.lr_schedulers
      ~InceptionTime.manual_backward
      ~InceptionTime.modules
      ~InceptionTime.mtia
      ~InceptionTime.named_buffers
      ~InceptionTime.named_children
      ~InceptionTime.named_modules
      ~InceptionTime.named_parameters
      ~InceptionTime.on_after_backward
      ~InceptionTime.on_after_batch_transfer
      ~InceptionTime.on_before_backward
      ~InceptionTime.on_before_batch_transfer
      ~InceptionTime.on_before_optimizer_step
      ~InceptionTime.on_before_zero_grad
      ~InceptionTime.on_fit_end
      ~InceptionTime.on_fit_start
      ~InceptionTime.on_load_checkpoint
      ~InceptionTime.on_predict_batch_end
      ~InceptionTime.on_predict_batch_start
      ~InceptionTime.on_predict_end
      ~InceptionTime.on_predict_epoch_end
      ~InceptionTime.on_predict_epoch_start
      ~InceptionTime.on_predict_model_eval
      ~InceptionTime.on_predict_start
      ~InceptionTime.on_save_checkpoint
      ~InceptionTime.on_test_batch_end
      ~InceptionTime.on_test_batch_start
      ~InceptionTime.on_test_end
      ~InceptionTime.on_test_epoch_end
      ~InceptionTime.on_test_epoch_start
      ~InceptionTime.on_test_model_eval
      ~InceptionTime.on_test_model_train
      ~InceptionTime.on_test_start
      ~InceptionTime.on_train_batch_end
      ~InceptionTime.on_train_batch_start
      ~InceptionTime.on_train_end
      ~InceptionTime.on_train_epoch_end
      ~InceptionTime.on_train_epoch_start
      ~InceptionTime.on_train_start
      ~InceptionTime.on_validation_batch_end
      ~InceptionTime.on_validation_batch_start
      ~InceptionTime.on_validation_end
      ~InceptionTime.on_validation_epoch_end
      ~InceptionTime.on_validation_epoch_start
      ~InceptionTime.on_validation_model_eval
      ~InceptionTime.on_validation_model_train
      ~InceptionTime.on_validation_model_zero_grad
      ~InceptionTime.on_validation_start
      ~InceptionTime.optimizer_step
      ~InceptionTime.optimizer_zero_grad
      ~InceptionTime.optimizers
      ~InceptionTime.parameters
      ~InceptionTime.predict_dataloader
      ~InceptionTime.predict_step
      ~InceptionTime.prepare_data
      ~InceptionTime.print
      ~InceptionTime.register_backward_hook
      ~InceptionTime.register_buffer
      ~InceptionTime.register_forward_hook
      ~InceptionTime.register_forward_pre_hook
      ~InceptionTime.register_full_backward_hook
      ~InceptionTime.register_full_backward_pre_hook
      ~InceptionTime.register_load_state_dict_post_hook
      ~InceptionTime.register_load_state_dict_pre_hook
      ~InceptionTime.register_module
      ~InceptionTime.register_parameter
      ~InceptionTime.register_state_dict_post_hook
      ~InceptionTime.register_state_dict_pre_hook
      ~InceptionTime.requires_grad_
      ~InceptionTime.save_hyperparameters
      ~InceptionTime.set_extra_state
      ~InceptionTime.set_submodule
      ~InceptionTime.setup
      ~InceptionTime.share_memory
      ~InceptionTime.state_dict
      ~InceptionTime.teardown
      ~InceptionTime.test_dataloader
      ~InceptionTime.test_step
      ~InceptionTime.to
      ~InceptionTime.to_empty
      ~InceptionTime.to_onnx
      ~InceptionTime.to_torchscript
      ~InceptionTime.toggle_optimizer
      ~InceptionTime.train
      ~InceptionTime.train_dataloader
      ~InceptionTime.training_step
      ~InceptionTime.transfer_batch_to_device
      ~InceptionTime.type
      ~InceptionTime.unfreeze
      ~InceptionTime.untoggle_optimizer
      ~InceptionTime.val_dataloader
      ~InceptionTime.validation_step
      ~InceptionTime.xpu
      ~InceptionTime.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~InceptionTime.CHECKPOINT_HYPER_PARAMS_KEY
      ~InceptionTime.CHECKPOINT_HYPER_PARAMS_NAME
      ~InceptionTime.CHECKPOINT_HYPER_PARAMS_TYPE
      ~InceptionTime.T_destination
      ~InceptionTime.automatic_optimization
      ~InceptionTime.call_super_init
      ~InceptionTime.current_epoch
      ~InceptionTime.device
      ~InceptionTime.device_mesh
      ~InceptionTime.dtype
      ~InceptionTime.dump_patches
      ~InceptionTime.example_input_array
      ~InceptionTime.fabric
      ~InceptionTime.global_rank
      ~InceptionTime.global_step
      ~InceptionTime.hparams
      ~InceptionTime.hparams_initial
      ~InceptionTime.local_rank
      ~InceptionTime.logger
      ~InceptionTime.loggers
      ~InceptionTime.on_gpu
      ~InceptionTime.strict_loading
      ~InceptionTime.trainer
      ~InceptionTime.training
   
   