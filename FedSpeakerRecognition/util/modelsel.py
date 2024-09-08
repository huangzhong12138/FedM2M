from network.fc import SpeakerIdetification
from network.models import AlexNet, PamapModel, lenet5v,VoiceprintLSTM,EcapaTdnn,TDNN,resnet18
import copy


def modelsel(args, device):
    if args.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid']:
        backbone = AlexNet(num_classes=args.num_classes).to(device)
    elif 'medmnist' in args.dataset:
        backbone = lenet5v().to(device)
    elif 'pamap' in args.dataset:
        backbone = PamapModel().to(device)
    else:
        # server_model = VoiceprintLSTM(input_size=40, hidden_size=64, num_layers=3, num_classes=args.num_classes).to(device)
        backbone = EcapaTdnn(channels = args.num_classes)
        # server_model= EcapaTdnn(channels = args.num_classes)
        # server_model = TDNN().to(device)
        # server_model = resnet18(args.num_classes).to(device)
    server_model = SpeakerIdetification(backbone=backbone,
                                      num_class=args.num_classes).to(device)

    client_weights = [1/args.n_clients for _ in range(args.n_clients)]  # 初始化客户端的权重
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights
