import torch
from thop import clever_format, profile

def calculate_macs(model, config, dataset=None):
    """
    The function calculate the multiply–accumulate operation (MACs) of the model given as input.

    Args:
        model: deep learning model to calculate the macs for
        config: config used to train the model
        dataset: dataset used to train the model

    Returns:

    """
    n_frames = int(
        (
            (config["feats"]["sample_rate"] * config["data"]["audio_max_len"])
            / config["feats"]["hop_length"]
        )
        + 1
    )
    # MACs 계산을 위한 더미 입력: [batch, channel, freq, time]
    input_size = [1, config["CRNN"]["n_input_ch"], config["feats"]["n_mels"], n_frames]
    input = torch.randn(input_size)

    if "use_embeddings" in config["CRNN"] and config["CRNN"]["use_embeddings"]:
        audio, label, padded_indxs, path, embeddings = dataset[0]
        embeddings = embeddings.repeat(1, 1, 1)
        macs, params = profile(model, inputs=(input, None, embeddings))
    else:
        macs, params = profile(model, inputs=(input,))

    macs, params = clever_format([macs, params], "%.3f")
    return macs, params



def count_parameters(model):
    """
    모델의 파라미터 수를 계산합니다.
    
    Args:
        model: PyTorch 모델
    
    Returns:
        total_params: 총 파라미터 수
        trainable_params: 훈련 가능한 파라미터 수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params
