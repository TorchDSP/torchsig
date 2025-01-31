torchsig.models.iq\_models.convit.convit1d.ConVit1DLightning
============================================================

.. currentmodule:: torchsig.models.iq_models.convit.convit1d

.. autoclass:: ConVit1DLightning
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~ConVit1DLightning.add_module
      ~ConVit1DLightning.all_gather
      ~ConVit1DLightning.apply
      ~ConVit1DLightning.backward
      ~ConVit1DLightning.bfloat16
      ~ConVit1DLightning.buffers
      ~ConVit1DLightning.children
      ~ConVit1DLightning.clip_gradients
      ~ConVit1DLightning.compile
      ~ConVit1DLightning.configure_callbacks
      ~ConVit1DLightning.configure_gradient_clipping
      ~ConVit1DLightning.configure_model
      ~ConVit1DLightning.configure_optimizers
      ~ConVit1DLightning.configure_sharded_model
      ~ConVit1DLightning.cpu
      ~ConVit1DLightning.cuda
      ~ConVit1DLightning.double
      ~ConVit1DLightning.eval
      ~ConVit1DLightning.extra_repr
      ~ConVit1DLightning.float
      ~ConVit1DLightning.forward
      ~ConVit1DLightning.freeze
      ~ConVit1DLightning.get_buffer
      ~ConVit1DLightning.get_extra_state
      ~ConVit1DLightning.get_parameter
      ~ConVit1DLightning.get_submodule
      ~ConVit1DLightning.half
      ~ConVit1DLightning.ipu
      ~ConVit1DLightning.load_from_checkpoint
      ~ConVit1DLightning.load_state_dict
      ~ConVit1DLightning.log
      ~ConVit1DLightning.log_dict
      ~ConVit1DLightning.lr_scheduler_step
      ~ConVit1DLightning.lr_schedulers
      ~ConVit1DLightning.manual_backward
      ~ConVit1DLightning.modules
      ~ConVit1DLightning.mtia
      ~ConVit1DLightning.named_buffers
      ~ConVit1DLightning.named_children
      ~ConVit1DLightning.named_modules
      ~ConVit1DLightning.named_parameters
      ~ConVit1DLightning.on_after_backward
      ~ConVit1DLightning.on_after_batch_transfer
      ~ConVit1DLightning.on_before_backward
      ~ConVit1DLightning.on_before_batch_transfer
      ~ConVit1DLightning.on_before_optimizer_step
      ~ConVit1DLightning.on_before_zero_grad
      ~ConVit1DLightning.on_fit_end
      ~ConVit1DLightning.on_fit_start
      ~ConVit1DLightning.on_load_checkpoint
      ~ConVit1DLightning.on_predict_batch_end
      ~ConVit1DLightning.on_predict_batch_start
      ~ConVit1DLightning.on_predict_end
      ~ConVit1DLightning.on_predict_epoch_end
      ~ConVit1DLightning.on_predict_epoch_start
      ~ConVit1DLightning.on_predict_model_eval
      ~ConVit1DLightning.on_predict_start
      ~ConVit1DLightning.on_save_checkpoint
      ~ConVit1DLightning.on_test_batch_end
      ~ConVit1DLightning.on_test_batch_start
      ~ConVit1DLightning.on_test_end
      ~ConVit1DLightning.on_test_epoch_end
      ~ConVit1DLightning.on_test_epoch_start
      ~ConVit1DLightning.on_test_model_eval
      ~ConVit1DLightning.on_test_model_train
      ~ConVit1DLightning.on_test_start
      ~ConVit1DLightning.on_train_batch_end
      ~ConVit1DLightning.on_train_batch_start
      ~ConVit1DLightning.on_train_end
      ~ConVit1DLightning.on_train_epoch_end
      ~ConVit1DLightning.on_train_epoch_start
      ~ConVit1DLightning.on_train_start
      ~ConVit1DLightning.on_validation_batch_end
      ~ConVit1DLightning.on_validation_batch_start
      ~ConVit1DLightning.on_validation_end
      ~ConVit1DLightning.on_validation_epoch_end
      ~ConVit1DLightning.on_validation_epoch_start
      ~ConVit1DLightning.on_validation_model_eval
      ~ConVit1DLightning.on_validation_model_train
      ~ConVit1DLightning.on_validation_model_zero_grad
      ~ConVit1DLightning.on_validation_start
      ~ConVit1DLightning.optimizer_step
      ~ConVit1DLightning.optimizer_zero_grad
      ~ConVit1DLightning.optimizers
      ~ConVit1DLightning.parameters
      ~ConVit1DLightning.predict_dataloader
      ~ConVit1DLightning.predict_step
      ~ConVit1DLightning.prepare_data
      ~ConVit1DLightning.print
      ~ConVit1DLightning.register_backward_hook
      ~ConVit1DLightning.register_buffer
      ~ConVit1DLightning.register_forward_hook
      ~ConVit1DLightning.register_forward_pre_hook
      ~ConVit1DLightning.register_full_backward_hook
      ~ConVit1DLightning.register_full_backward_pre_hook
      ~ConVit1DLightning.register_load_state_dict_post_hook
      ~ConVit1DLightning.register_load_state_dict_pre_hook
      ~ConVit1DLightning.register_module
      ~ConVit1DLightning.register_parameter
      ~ConVit1DLightning.register_state_dict_post_hook
      ~ConVit1DLightning.register_state_dict_pre_hook
      ~ConVit1DLightning.requires_grad_
      ~ConVit1DLightning.save_hyperparameters
      ~ConVit1DLightning.set_extra_state
      ~ConVit1DLightning.set_submodule
      ~ConVit1DLightning.setup
      ~ConVit1DLightning.share_memory
      ~ConVit1DLightning.state_dict
      ~ConVit1DLightning.teardown
      ~ConVit1DLightning.test_dataloader
      ~ConVit1DLightning.test_step
      ~ConVit1DLightning.to
      ~ConVit1DLightning.to_empty
      ~ConVit1DLightning.to_onnx
      ~ConVit1DLightning.to_torchscript
      ~ConVit1DLightning.toggle_optimizer
      ~ConVit1DLightning.train
      ~ConVit1DLightning.train_dataloader
      ~ConVit1DLightning.training_step
      ~ConVit1DLightning.transfer_batch_to_device
      ~ConVit1DLightning.type
      ~ConVit1DLightning.unfreeze
      ~ConVit1DLightning.untoggle_optimizer
      ~ConVit1DLightning.val_dataloader
      ~ConVit1DLightning.validation_step
      ~ConVit1DLightning.xpu
      ~ConVit1DLightning.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~ConVit1DLightning.CHECKPOINT_HYPER_PARAMS_KEY
      ~ConVit1DLightning.CHECKPOINT_HYPER_PARAMS_NAME
      ~ConVit1DLightning.CHECKPOINT_HYPER_PARAMS_TYPE
      ~ConVit1DLightning.T_destination
      ~ConVit1DLightning.automatic_optimization
      ~ConVit1DLightning.call_super_init
      ~ConVit1DLightning.current_epoch
      ~ConVit1DLightning.device
      ~ConVit1DLightning.device_mesh
      ~ConVit1DLightning.dtype
      ~ConVit1DLightning.dump_patches
      ~ConVit1DLightning.example_input_array
      ~ConVit1DLightning.fabric
      ~ConVit1DLightning.global_rank
      ~ConVit1DLightning.global_step
      ~ConVit1DLightning.hparams
      ~ConVit1DLightning.hparams_initial
      ~ConVit1DLightning.local_rank
      ~ConVit1DLightning.logger
      ~ConVit1DLightning.loggers
      ~ConVit1DLightning.on_gpu
      ~ConVit1DLightning.strict_loading
      ~ConVit1DLightning.trainer
      ~ConVit1DLightning.training
   
   