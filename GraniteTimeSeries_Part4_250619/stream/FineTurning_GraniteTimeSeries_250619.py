#---------------------------------------------------------------
# IBM Granite Time Series ( Tiny Time Mixer )
# HugingFace Granite Time Series �𗘗p
#     "ibm-granite/granite-timeseries-ttm-r2
#     "512-96-ft-r2.1", 
#     FineTurning Model���g���ė\��
#---------------------------------------------------------------
#---------------------------------------------------------------
# A. ���C�u���������p�[�g
#---------------------------------------------------------------
# Modeler�p���C�u����
import modelerpy

# �e�폈���p
import numpy as np
import pandas as pd

# Finetuning���f���ۑ��p
import math
import os

# ���f���쐬�p - Finetuning
# pytorch
import torch

# �œK����@�̈�ŁAAdam �� weight decay�i�d�݂̌����j ������������
# �⑫: �ʏ��Adam����L2�����������܂������Ȃ����Ƃ����� �� ������C�������̂�AdamW
from torch.optim import AdamW 

# �X�P�W���[��
# �w�K���iLearning Rate�j�� 1�T�C�N���̌`�Œ�������X�P�W���[��
from torch.optim.lr_scheduler import OneCycleLR # �ŏ�������聨���Ց������Ō�������c�Ɗw�K���𓮓I�ɒ���

# �����I���iEarly Stopping�j �p�̃R�[���o�b�N
# �g���[�j���O�����N���X
# Trainer �ɓn���ݒ�i�w�K���A�o�b�`�T�C�Y�A�o�͐�Ȃǁj��ێ�����N���X
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

# IBM Granite TTM 
from tsfm_public import (
    TimeSeriesForecastingPipeline,    # ���n��\���̃p�C�v���C��
    TimeSeriesPreprocessor,           # ���̓f�[�^�̐��`�E�X�P�[�����O
    TinyTimeMixerForPrediction,       # ���n��\�����f��
    TrackingCallback,                 # �w�K���Ƀ��O���������A�ǉ��ŉ������L�^���邽�߂� �J�X�^��Callback
    count_parameters,                 # ���f���̑��p�����[�^���𐔂��ďo�͂��Ă����֐�
    get_datasets,                     # �O�����ς݂�DataFrame���A���f���������� Dataset�`���i�w�K/����/�e�X�g�j �ɕϊ�����֐�
)
# ���̓f�[�^��train/valid/test �Ɏ����ŕ������Ă����֐�
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits 

#---------------------------------------------------------------
# B. �J������`
#---------------------------------------------------------------
# �^�C���X�^���v�J����
timestamp_column = "time"
# �\���ΏۃJ������
target_column = "total load actual"
# ���f���̗\���l�i�[�J������
prediciton_column = target_column + "_prediction"
# Modeler�p�\���l�i�[�J������
predictField = "$GTS-" + target_column

#-----------------------------------------------
# C. �g�p���f���p�p�����[�^�ݒ�
#-----------------------------------------------
# ���f�����u�ߋ����X�e�b�v���̃f�[�^�����āv�w�K�E�\�����邩�i�����ł͉ߋ�512���ԕ��j
context_length = 512  # the max context length for the 512-96 model
# ���f�����u�������X�e�b�v��܂ŗ\���v���邩�i�����ł�96���Ԑ恁4�����j
prediction_length = 96  # the max forecast length for the 512-96 model
# �g�p���f���o�[�W����
TTM_MODEL_REVISION = "512-96-ft-r2.1"

#-----------------------------------------------
# D. �n�C�p�[�p�����[�^�ݒ�
#-----------------------------------------------
learning_rate: float = 0.001 # learning_rate: �w�K�̑���
num_epochs: int = 20         # num_epochs: �ő� 20 �G�|�b�N�܂Ŋw�K
patience: int = 5            # patience: ���x�����P���Ȃ���Ԃ� 5 �G�|�b�N�������瑁���I��
batch_size: int = 32         # batch_size: ��x�ɉ��̎��n��X���C�X���w�K���邩�i32�j


#-----------------------------------------------
# E.�f�[�^���f����`
#-----------------------------------------------
if modelerpy.isComputeDataModelOnly():
    #�f�[�^���f���擾
    modelerDataModel = modelerpy.getDataModel()

    #���f���p�ϐ��t�B�[���h��ǉ�
    #predictField - �������i�[�\��
    modelerDataModel.addField(modelerpy.Field( predictField, "real", measure="continuous"))

    #�C�������f�[�^���f����Modeler�֖߂��܂��B
    modelerpy.setOutputDataModel(modelerDataModel)

#-----------------------------------------------
# F.���n�񃂃f���p�\�h
# 1. �f�[�^�̓���
# 2. ���f���p�f�[�^����
# 3. ���f�����s�ݒ�
# 4. ���f���쐬 - FineTurning���s
# 5. Modeler�ւ̃f�[�^�o�͏���
#-----------------------------------------------
else:

    #---------------------------------------------------------------
    # 1. �f�[�^���̓p�[�g
    #---------------------------------------------------------------
    #Modeler����Pandas�Ńf�[�^�����
    modelerData = modelerpy.readPandasDataframe()
    input_df = modelerData

    # ���n��J����"time"���^�C���X�^���v�^�ɕύX
    input_df[ timestamp_column ] = pd.to_datetime(input_df[ timestamp_column ])

    #---------------------------------------------------------------
    # 2. ���f���p�f�[�^�����p�[�g
    #---------------------------------------------------------------
    #-----------------------------------------------
    # �J�������[���ݒ� 
    #-----------------------------------------------
    # �\���ΏۃJ������`
    target_columns = [ target_column ] #���X�g�ɂ���

    # �J�������[���ݒ�
    # �����ϐ��ɂ́A�e�����̋C�ۃf�[�^�̒����l���w��
    column_specifiers = {
        "timestamp_column": timestamp_column, # �^�C���X�^���v�J����
        "id_columns": [],                     # �����n������ʂ��邽�߂�ID�i�����1�n��Ȃ̂ŋ�j
        "target_columns": target_columns,     # �\���ΏۃJ����
        "control_columns": [                  # �R���g���[���ϐ��i�C�ۃf�[�^�Ȃǁj
            "temp_MA24H_Median",
            "temp_min_MA24H_Median",
            "temp_max_MA24H_Median",
            "pressure_MA24H_Median",
            "wind_speed_MA24H_Median",
            "wind_deg_MA24H_Median",
            "humidity_MA24H_Median",
            "rain_1h_MA24H_Median",
            "rain_3h_MA24H_Median",
            "clouds_all_MA24H_Median",
        ], 
    }


    #-----------------------------------------------
    # �g�p�f�[�^�ݒ�
    #-----------------------------------------------
    # �f�[�^�Z�b�g�̂����A10%�������g���ăg���[�j���O�E�]�����s���ݒ�
    fewshot_fraction = 0.1

    # �f�[�^�����̐ݒ� - �w�K�p 60% , �e�X�g�p 20% , �c�� 20%
    split_config = {"train": 0.6, "test": 0.2}

    # Finetuning���̃��O�A���f���Ȃǂ̕ۑ��t�H���_
    OUT_DIR = "C:\\Qiita\\202505\\ttm_results"

    # GPU�iCUDA�j���g����Ύg���A�������CPU�B
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #-----------------------------------------------
    # �f�[�^�����u�P���E���؁E�e�X�g�v
    #-----------------------------------------------
    # input_df �� context_length ���l�����Ȃ���u�P���E���؁E�e�X�g�v�ɕ���
    train_df, valid_df, test_df = prepare_data_splits(input_df, context_length=context_length, split_config=split_config)

    #---------------------------------------------------------------
    # 3. ���f�����s�ݒ�p�[�g
    #---------------------------------------------------------------
    #-----------------------------------------------
    # ���f���p Preprocessor �̏������Ɗw�K
    #-----------------------------------------------
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,                    # �J������`
        context_length=context_length,          # �w�K�Ɏg�p���郌�R�[�h��(�����ł�512)
        prediction_length=prediction_length,    # �\�����郌�R�[�h��(96)
        scaling=True,                           # �X�P�[�����O�i�W�����␳�K���j
        encode_categorical=False,               # �J�e�S���ϐ��i������Ȃǁj�𐔒l�ɃG���R�[�h���邩�ǂ����B���l�f�[�^�݂̂̂���False
        scaler_type="standard",                 # ����0�E�W���΍�1�ɕW�����iStandardScaler���g���C���[�W)
    )

    #-----------------------------------------------
    # ���f���p�����[�^�ݒ�
    #-----------------------------------------------
    # �����_��SeeD�̐ݒ� - 42�͂��܂��Ȃ�
    set_seed(42)

    # Granite Time Series���f���ݒ�
    finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",                   # Hugging Face ��� IBM��Granite�V���[�Y ���n�񃂃f���iTinyTimeMixer�j
        num_input_channels=tsp.num_input_channels,                 # ���͂̃`�����l�����i= ���f���ɓn�������ʂ̐��j�@tsp�Ŏ����Ōv�Z����Ă�B
        prediction_channel_indices=tsp.prediction_channel_indices, # �\���Ώۂ̎w�� tsp�Ōv�Z����Ă���B
        exogenous_channel_indices=tsp.exogenous_channel_indices,   # �O���ϐ��i�G�N�\�W�F�j�A�X�ϐ��j�v�̈ʒu���w��Btsp�Ōv�Z����Ă���B
        fcm_use_mixer=True,                                        # FCM = Forecast Channel Mixing �����̓����ʁi�`���l���j�������Ȃ��玞�n��̖�����\�����邵���݂�L��������B
        fcm_context_length=10,                                     # FCM�ɓn���u�ߋ��̎Q�ƃE�B���h�E�v�̒����B10�X�e�b�v���̃f�[�^�����Ė������l����A�Ƃ����ݒ�B
        enable_forecast_channel_mixing=True,                       # �`���l���Ԃ� Mixing �� Decoder ���ł��g�����ǂ����B�L���ɂ���ƁA�\�����o���Ƃ��ɂ������ʓ��m�������āA��蕡�G�ȃp�^�[���𑨂��邱�Ƃ��ł���B
        decoder_mode="mix_channel",                                # Decoder�i�\���𐶐����鑤�j�ł̋����ݒ�B"mix_channel" �ɂ���ƁA�`�����l���Ԃ̃~�L�V���O��L���������\���ɂȂ�
    )

    #-----------------------------------------------
    # ���O�w�K���X�V�������p���� (backborne)�ݒ�
    #-----------------------------------------------
    # **�ubackbone�v�����̏d�݂́A���O�w�K���ꂽ�܂܌Œ肵�āA�w�K���ɂ͍X�V���Ȃ��i��������j**�Ƃ����Ӗ�
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False

    #-----------------------------------------------
    # �f�[�^�Z�b�g�쐬(Preprocessor�̓��e�𔽉f)
    #-----------------------------------------------
    # �f�[�^�Z�b�g�̍쐬 (split_config 60:20:20)
    # �w�K�E���؁E�e�X�g�p��PyTorch Dataset �ɕϊ�
    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp,                                  # ���O��`����Preprocessor
        input_df,
        split_config,                         # 60:20:20
        fewshot_fraction=fewshot_fraction,    # 0.1
        fewshot_location="first",             # few-shot�Ɏg���f�[�^�̎����B�ufirst�v�Ȃ�A�f�[�^�̍ŏ�����few-shot�T���v�������B�i"random"������j
        use_frequency_token=finetune_forecast_model.config.resolution_prefix_tuning, # ���g�����i���Ԃ̗��x���A'1H'��'1D'�𖄂ߍ��ނ��H)�B�����ł̓��f���̎��O�ݒ�����̂܂܎g���ݒ�
    )

    #-----------------------------------------------
    # Fine-Tuning�̃p�����[�^�ݒ�
    #-----------------------------------------------
    # Trainer �̊w�K�ݒ�iTrainingArguments�j
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"), # ���f���o�͐�f�B���N�g��
        overwrite_output_dir=True,                  # �㏑
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,                               # �e�G�|�b�N���ƂɌ��؂��s��
        eval_strategy="epoch",                      # 1�G�|�b�N���Ƃɕ]��
        per_device_train_batch_size=batch_size,     
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,                   # CPU�Ȃ� 0 ���� (�}���`�X���b�h������)
        report_to=None,
        save_strategy="epoch",                      # 1�G�|�b�N���ƂɃ��f���ۑ��i������ save_total_limit=1 �Ȃ̂ōŐV�����c���j
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,                # �ł��ǂ������G�|�b�N�̃��f����ۑ�
        metric_for_best_model="eval_loss",          # �]���w�W�͑����֐��i�����������ǂ��j
        greater_is_better=False,                    # For loss
        use_cpu=device != "cuda",                   # GPU ���g���Ȃ��Ƃ��� CPU �Ŏ��s
    )

    #-----------------------------------------------
    # �����I���̐ݒ�
    #-----------------------------------------------
    # EarlyStopping�i�����I���̐ݒ�j
    # �]������ (eval_loss) �̉��P�� 0.001 ������ 5 �G�|�b�N �����Ǝ����I��
    # �� ���ʂȊw�K��h���ŃI�[�o�[�t�B�b�e�B���O������I
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=patience,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.001,  # Minimum improvement required to consider as improvement
    )

    # ���O�Ȃǂ̃J�X�^���ݒ�
    tracking_callback = TrackingCallback()

    #-----------------------------------------------
    # Optimizer / �X�P�W���[���ݒ�
    #-----------------------------------------------
    # Optimizer�i�œK����@�j�� Scheduler�i�w�K���X�P�W���[���[�j�̐ݒ�
    # AdamW�F��Ԃ̍œK����@�i�d�݌����ɋ����j
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)

    # OneCycleLR�F�ŏ�������聨���Ց������Ō�������c�Ɗw�K���𓮓I�ɒ���
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    #-----------------------------------------------
    # 4. ���f���쐬 - FineTurning���s�p�[�g
    #-----------------------------------------------
    #-----------------------------------------------
    # FineTurning�̎��s
    #-----------------------------------------------
    # Trainer �C���X�^���X�쐬
    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,                          # FineTune���f��
        args=finetune_forecast_args,                            # Trainer �̊w�K�ݒ�
        train_dataset=train_dataset,                            # �w�K�p�f�[�^�Z�b�g
        eval_dataset=valid_dataset,                             # ���ؗp�f�[�^�Z�b�g
        callbacks=[early_stopping_callback, tracking_callback], # EarlyStopping�i�����I���̐ݒ�j
        optimizers=(optimizer, scheduler),                      # Optimizer�i�œK����@�j�� Scheduler�i�w�K���X�P�W���[���[�j
    )

    # �w�K�X�^�[�g
    finetune_forecast_trainer.train()

    #-----------------------------------------------
    # ���f��Pipeline�̍쐬�E���s
    #-----------------------------------------------
    # �\���p�C�v���C���̍쐬�Ǝ��s
    pipeline = TimeSeriesForecastingPipeline(
        finetune_forecast_model,        #  fewshot - finetune���f��
        device=device,                  #  GPU or CPU.
        feature_extractor=tsp,          #  Preprocessor
        batch_size=batch_size,
    )

    # Make a forecast on the target column given the input data.
    finetune_forecast = pipeline(input_df)

    #------------------------------------------------------------
    # 5. Modeler�ɖ߂����߂̃f�[�^���H�p�[�g
    # 1. �ߋ��f�[�^�ɑ΂���\���l�̎擾�����̓f�[�^�Ɍ���
    #    �߂�l�̓������l�����āA���X�g�̐擪�̒l������̃��R�[�h�֔z�u
    # 2. �Ō�̃��R�[�h����96���R�[�h��܂ł̗\�����擾
    # 3. �ŏI���R�[�h��薢���̒l��1.�̃f�[�^�Ɍ��������AModeler�ɖ߂�
    #------------------------------------------------------------
    #=====================================
    # Step 1: �ߋ���1�X�e�b�v��\�����i�[
    # �ߋ��̎��ђl�ɑ΂��Ẵ��f���̗\���l�����߂Ă���
    #=====================================
    # ���f���̖߂�l���R�s�[
    past_pred_df = finetune_forecast.copy()

    # ndarray����1�ڂ̒l���擾����֐�
    def get_first_val(x):
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            return x[0]
        return np.nan

    # �\���l��1�X�e�b�v�悾�����o���i���X�g��1�ڂ��擾)
    past_pred_df[ predictField ] = (
        past_pred_df[ prediciton_column ].apply(get_first_val)
    )

    # 1�X�e�b�v��Ȃ̂ŁA�u1���Ԍ�v�ɂ��炷�i���̃��R�[�h�Ɋi�[�j
    past_pred_df[ predictField ] = past_pred_df[ predictField ].shift(1)

    # input_df �Ƀ}�[�W����itime�Ō����Ainner join ��OK�j
    merged_df = pd.merge(input_df, past_pred_df[[timestamp_column, predictField]], on=timestamp_column, how="left")
    #=====================================
    # Step 2: 96���Ԑ�̖�����\�����鏈��
    # �V�����f�[�^�t���[���ɖ����̒l���i�[
    #=====================================
    # ���͂̍Ō��512�������擾�i512��context_length�j
    input_tail = input_df.tail(context_length)

    # ����96���ԕ��̋�� DataFrame ���쐬
    last_time = input_tail[timestamp_column].iloc[-1]
    future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=prediction_length, freq="h")

    # ���DataFrame�����iNaN������j
    future_df = pd.DataFrame({timestamp_column: future_times})

    # ����+������DataFrame������ė\���i�O�����킪�����ŕ⊮�ȂǍs���j
    input_for_forecast = pd.concat([input_tail, future_df], ignore_index=True)

    # �\�������s
    future_forecast = pipeline(input_for_forecast)
 
    # �����\������������؂�o���ĐV����DataFrame��
    # �e�s�� total load actual_prediction �� 96�̗\���l�����������X�g�Ȃ̂ŁA
    # �ŏ��̍s�����g���� 1���R�[�h�ɓW�J
    pred_list = future_forecast[ prediciton_column ].iloc[0]

    future_df = pd.DataFrame({
        timestamp_column: future_times,
        predictField: pred_list,
    })

    #=====================================
    # Step 3: �ߋ��f�[�^�̗\���l�Ɩ����̒l����������Modeler�ɖ߂�
    # �V�����f�[�^�t���[���ɖ����̒l���i�[
    #=====================================
    # ���̃J�����itotal load actual �Ȃǁj�� input_df �Ɠ����\���ɂ��邽�߁ANaN �Ŗ��߂�
    for col in merged_df.columns:
        if col not in future_df.columns:
            future_df[col] = np.nan

    # �J�������� merged_df �ɍ��킹�ĕ��בւ�
    future_df = future_df[merged_df.columns]

    # merged_df + future_df ���c�Ɍ���
    input_df_with_pred = pd.concat([merged_df, future_df], ignore_index=True)

    # ���ʊm�F
    print(input_df_with_pred.tail(100))  # �Ō��100�s��\��

    # Modeler�Ƀf�[�^��߂�
    modelerpy.writePandasDataframe(input_df_with_pred)

