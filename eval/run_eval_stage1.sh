CUDA_VISIBLE_DEVICES=2 \
bash eval/run_metrics.sh \
  --real_videos_dir /home/work/.local/HDTF/HDTF_original_testset_81frames/videos_cfr \
  --fake_videos_dir /home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/inference/stableavatar_ref_multistep_stage1/step-6900/hdtf_short/composited_latentsync_with_audio \
  --shape_predictor_path /home/work/.local/LatentSync/shape_predictor_68_face_landmarks.dat \
  --output_dir /home/work/.local/eval_metrics/logs/stableavatar_ref_multistep_stage1_step-6900 \
  --log_path /home/work/.local/eval_metrics/logs/stableavatar_ref_multistep_stage1_step-6900/all_metrics.log \
  --fallback_detection_confidence 0.2 \
  --all \