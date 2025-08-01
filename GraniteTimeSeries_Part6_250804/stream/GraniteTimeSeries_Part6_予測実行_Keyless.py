#-------------------------------------------------
# IBM Granite Time Series ( Tiny Time Mixer )
# Watsonx.ai��� Granite Time Series �𗘗p
# Modeler�A�g - �\���l�擾�V���^�b�N�X
#-------------------------------------------------
#-----------------------------------------------
# ���C�u���������p�[�g
# �@�E�f�[�^���f����`���̂��߂ɕK�v�Œ���𓱓�
#-----------------------------------------------
# Modeler�p���C�u����
import modelerpy

# ���ׂĂ̌x���𖳎�
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------
# �J������`
#-----------------------------------------------
# �^�C���X�^���v�J������`
timestamp_column = "time"
# �\���ΏۃJ������
target_column = "total load actual"

#-----------------------------------------------
# �f�[�^���f����`�p�[�g
#-----------------------------------------------
if modelerpy.isComputeDataModelOnly():
    #�f�[�^���f���擾
    modelerDataModel = modelerpy.getDataModel()
    #���ɕύX���Ȃ�
    modelerpy.setOutputDataModel(modelerDataModel)

#-----------------------------------------------
# ���n��\���l�擾�p�\�h
#-----------------------------------------------
else:

    #-----------------------------------------------
    # ���f���쐬�p���C�u�����̃C���|�[�g
    #-----------------------------------------------
    # Pandas & Numpy
    import numpy as np
    import pandas as pd

    # IBM Watsonx.ai ���C�u����
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai import Credentials

    # Granite Time Series Forcasting on IBM Watsonx.ai ���C�u����
    from ibm_watsonx_ai.foundation_models import TSModelInference
    from ibm_watsonx_ai.foundation_models.schema import TSForecastParameters

    #-----------------------------------------------
    # �f�[�^���� 
    #-----------------------------------------------
    #Modeler����Pandas�Ńf�[�^�����
    modelerData = modelerpy.readPandasDataframe()
    input_df = modelerData

    #-----------------------------------------------
    # API Key �Ȃǂ̏���
    #-----------------------------------------------
    WATSONX_APIKEY = "YOUR API KEY"
    WATSONX_PROJECT_ID = "YOUR PROJECT ID"
    WATSONX_URL = "YOUR ENDPOINT URL"

    #-----------------------------------------------
    # Watsonx.ai �����^�C���ւ̐ڑ����ݒ�
    #-----------------------------------------------
    # watsonx.ai�����^�C���ւ̐ڑ������`
    credentials = Credentials(
        url = WATSONX_URL,
        api_key = WATSONX_APIKEY,
    )

    # �ڑ����̐ݒ�
    # �N���f���V�����̃Z�b�g
    client = APIClient(credentials)
    # �v���W�F�N�g�̃Z�b�g
    client.set.default_project( WATSONX_PROJECT_ID )

    #-----------------------------------------------
    # �w�K�f�[�^���R�[�h���A�\�����R�[�h���ݒ�
    #-----------------------------------------------
    # ���f�����u�ߋ����X�e�b�v���̃f�[�^�����āv�w�K�E�\�����邩�i�����ł͉ߋ�512���ԕ��j
    context_length = 512  # the max context length for the 512-96 model
    # �\���Ώہi"total load actual"�j
    prediction_length = 96  # the max forecast length for the 512-96 model

    #-----------------------------------------------
    # ���f���J������`
    #-----------------------------------------------
    # �\���ΏۃJ���������X�g�Œ�`
    target_columns = [ target_column ]

    # ���n��J����"time"�𕶎���^�ɕύX
    # Watsonx �̗����ł́AHTTP ���N�G�X�g��ʂ��ă��f���Ƀf�[�^��n�����߁A
    # JSON�`���ɕϊ��\�ȃf�[�^�i������A���l�A�z��Ȃǁj�����n���܂���B
    input_df[ timestamp_column ] = input_df[ timestamp_column ] .astype(str)

    #-----------------------------------------------
    # ���f���̗\���E���ؗp�f�[�^�̏���
    #-----------------------------------------------
    # �\���p�f�[�^ - �Ōォ��512���R�[�h���g��
    future_data = input_df.iloc[-context_length:,]

    #-----------------------------------------------
    # ���f���p�����[�^��`
    #-----------------------------------------------
    forecasting_params = TSForecastParameters(
        id_columns=[],                       # �����n������ʂ��邽�߂�ID�i�����1�n��Ȃ̂ŋ�j
        timestamp_column=timestamp_column,   # ����
        freq="1h",                           # ���ԒP�� - 1���Ԃ��Ƃ̃f�[�^
        target_columns=target_columns,       # �\���Ώہi"total load actual"�j
        prediction_length=prediction_length, # �\�����钷�� 
    )

    #-----------------------------------------------
    # Watsonx.ai���f���̃C���X�^���X��
    #-----------------------------------------------
    #���f���̎w�� - 512���R�[�h���g��96���R�[�h���\�����郂�f�����w��
    ts_model_id = client.foundation_models.TimeSeriesModels.GRANITE_TTM_512_96_R2
    # ���f���C���X�^���X�̏������Ɛݒ� - APIKEY��Project ID�������Ŏg�p
    ts_model = TSModelInference(model_id=ts_model_id, api_client=client)

    #-----------------------------------------------
    # ���f�����s
    #-----------------------------------------------
    # �\���p�f�[�^(�Ōォ��512���R�[�h)�Ń��f�����s
    results = ts_model.forecast(data=future_data, params=forecasting_params)['results'][0]
    # �\�����ʂ��f�[�^�t���[���Ɏ�荞�� - �\���l 96 ���R�[�h���i�[����Ă���
    watsonx_gts_forecast = pd.DataFrame(results)

    #-----------------------------------------------
    # ���̓f�[�^�ɗ\���l��ǉ����Ė߂�
    #-----------------------------------------------
    # ���ꂼ���time�̃J������time stamp�ɕϊ����Ă���ǉ�����
    modelerData[ timestamp_column ] = pd.to_datetime(modelerData[ timestamp_column ])
    watsonx_gts_forecast[ timestamp_column ] = pd.to_datetime(watsonx_gts_forecast[ timestamp_column ])

    # �ǉ�
    df_combined = pd.concat([modelerData, watsonx_gts_forecast], ignore_index=True)
  
    # Modeler�Ƀf�[�^��߂�
    modelerpy.writePandasDataframe(df_combined)