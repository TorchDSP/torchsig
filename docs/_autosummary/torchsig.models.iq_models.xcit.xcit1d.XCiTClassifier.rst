torchsig.models.iq\_models.xcit.xcit1d.XCiTClassifier
=====================================================

.. currentmodule:: torchsig.models.iq_models.xcit.xcit1d

.. autoclass:: XCiTClassifier
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~XCiTClassifier.add_module
      ~XCiTClassifier.all_gather
      ~XCiTClassifier.apply
      ~XCiTClassifier.backward
      ~XCiTClassifier.bfloat16
      ~XCiTClassifier.buffers
      ~XCiTClassifier.children
      ~XCiTClassifier.clip_gradients
      ~XCiTClassifier.compile
      ~XCiTClassifier.configure_callbacks
      ~XCiTClassifier.configure_gradient_clipping
      ~XCiTClassifier.configure_model
      ~XCiTClassifier.configure_optimizers
      ~XCiTClassifier.configure_sharded_model
      ~XCiTClassifier.cpu
      ~XCiTClassifier.cuda
      ~XCiTClassifier.double
      ~XCiTClassifier.eval
      ~XCiTClassifier.extra_repr
      ~XCiTClassifier.float
      ~XCiTClassifier.forward
      ~XCiTClassifier.freeze
      ~XCiTClassifier.get_buffer
      ~XCiTClassifier.get_extra_state
      ~XCiTClassifier.get_parameter
      ~XCiTClassifier.get_submodule
      ~XCiTClassifier.half
      ~XCiTClassifier.ipu
      ~XCiTClassifier.load_from_checkpoint
      ~XCiTClassifier.load_state_dict
      ~XCiTClassifier.log
      ~XCiTClassifier.log_dict
      ~XCiTClassifier.lr_scheduler_step
      ~XCiTClassifier.lr_schedulers
      ~XCiTClassifier.manual_backward
      ~XCiTClassifier.modules
      ~XCiTClassifier.named_buffers
      ~XCiTClassifier.named_children
      ~XCiTClassifier.named_modules
      ~XCiTClassifier.named_parameters
      ~XCiTClassifier.on_after_backward
      ~XCiTClassifier.on_after_batch_transfer
      ~XCiTClassifier.on_before_backward
      ~XCiTClassifier.on_before_batch_transfer
      ~XCiTClassifier.on_before_optimizer_step
      ~XCiTClassifier.on_before_zero_grad
      ~XCiTClassifier.on_fit_end
      ~XCiTClassifier.on_fit_start
      ~XCiTClassifier.on_load_checkpoint
      ~XCiTClassifier.on_predict_batch_end
      ~XCiTClassifier.on_predict_batch_start
      ~XCiTClassifier.on_predict_end
      ~XCiTClassifier.on_predict_epoch_end
      ~XCiTClassifier.on_predict_epoch_start
      ~XCiTClassifier.on_predict_model_eval
      ~XCiTClassifier.on_predict_start
      ~XCiTClassifier.on_save_checkpoint
      ~XCiTClassifier.on_test_batch_end
      ~XCiTClassifier.on_test_batch_start
      ~XCiTClassifier.on_test_end
      ~XCiTClassifier.on_test_epoch_end
      ~XCiTClassifier.on_test_epoch_start
      ~XCiTClassifier.on_test_model_eval
      ~XCiTClassifier.on_test_model_train
      ~XCiTClassifier.on_test_start
      ~XCiTClassifier.on_train_batch_end
      ~XCiTClassifier.on_train_batch_start
      ~XCiTClassifier.on_train_end
      ~XCiTClassifier.on_train_epoch_end
      ~XCiTClassifier.on_train_epoch_start
      ~XCiTClassifier.on_train_start
      ~XCiTClassifier.on_validation_batch_end
      ~XCiTClassifier.on_validation_batch_start
      ~XCiTClassifier.on_validation_end
      ~XCiTClassifier.on_validation_epoch_end
      ~XCiTClassifier.on_validation_epoch_start
      ~XCiTClassifier.on_validation_model_eval
      ~XCiTClassifier.on_validation_model_train
      ~XCiTClassifier.on_validation_model_zero_grad
      ~XCiTClassifier.on_validation_start
      ~XCiTClassifier.optimizer_step
      ~XCiTClassifier.optimizer_zero_grad
      ~XCiTClassifier.optimizers
      ~XCiTClassifier.parameters
      ~XCiTClassifier.predict_dataloader
      ~XCiTClassifier.predict_step
      ~XCiTClassifier.prepare_data
      ~XCiTClassifier.print
      ~XCiTClassifier.register_backward_hook
      ~XCiTClassifier.register_buffer
      ~XCiTClassifier.register_forward_hook
      ~XCiTClassifier.register_forward_pre_hook
      ~XCiTClassifier.register_full_backward_hook
      ~XCiTClassifier.register_full_backward_pre_hook
      ~XCiTClassifier.register_load_state_dict_post_hook
      ~XCiTClassifier.register_module
      ~XCiTClassifier.register_parameter
      ~XCiTClassifier.register_state_dict_pre_hook
      ~XCiTClassifier.requires_grad_
      ~XCiTClassifier.save_hyperparameters
      ~XCiTClassifier.set_extra_state
      ~XCiTClassifier.setup
      ~XCiTClassifier.share_memory
      ~XCiTClassifier.state_dict
      ~XCiTClassifier.teardown
      ~XCiTClassifier.test_dataloader
      ~XCiTClassifier.test_step
      ~XCiTClassifier.to
      ~XCiTClassifier.to_empty
      ~XCiTClassifier.to_onnx
      ~XCiTClassifier.to_torchscript
      ~XCiTClassifier.toggle_optimizer
      ~XCiTClassifier.train
      ~XCiTClassifier.train_dataloader
      ~XCiTClassifier.training_step
      ~XCiTClassifier.transfer_batch_to_device
      ~XCiTClassifier.type
      ~XCiTClassifier.unfreeze
      ~XCiTClassifier.untoggle_optimizer
      ~XCiTClassifier.val_dataloader
      ~XCiTClassifier.validation_step
      ~XCiTClassifier.xpu
      ~XCiTClassifier.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~XCiTClassifier.CHECKPOINT_HYPER_PARAMS_KEY
      ~XCiTClassifier.CHECKPOINT_HYPER_PARAMS_NAME
      ~XCiTClassifier.CHECKPOINT_HYPER_PARAMS_TYPE
      ~XCiTClassifier.T_destination
      ~XCiTClassifier.automatic_optimization
      ~XCiTClassifier.call_super_init
      ~XCiTClassifier.current_epoch
      ~XCiTClassifier.device
      ~XCiTClassifier.device_mesh
      ~XCiTClassifier.dtype
      ~XCiTClassifier.dump_patches
      ~XCiTClassifier.example_input_array
      ~XCiTClassifier.fabric
      ~XCiTClassifier.global_rank
      ~XCiTClassifier.global_step
      ~XCiTClassifier.hparams
      ~XCiTClassifier.hparams_initial
      ~XCiTClassifier.local_rank
      ~XCiTClassifier.logger
      ~XCiTClassifier.loggers
      ~XCiTClassifier.on_gpu
      ~XCiTClassifier.strict_loading
      ~XCiTClassifier.trainer
      ~XCiTClassifier.training
   
   