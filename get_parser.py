import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Finetune or evaluate a Resnet model on Imagewoof using the "
            "from a checkpoint."
        )
    )
    parser.add_argument(
        "--do-tune",
        action="store_true",
        help="Run training loop.",
    )
    parser.add_argument(
        "--do-tune-eval",
        action="store_true",
        help="Do evaluation after each training epoch.",
    )
    parser.add_argument(
        "--tune-batch-size",
        type=int,
        required=False,
        default=128,
        help="Training data loader batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        required=False,
        default=32,
        help="Validation data loader batch size",
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        type=str,
        required=True,
        help="Path to directory containing validation/test set",
    )
    parser.add_argument(
        "--resnet-size",
        type=int,
        required=False,
        default=18,
        help="Size of ResNet model. Supported models are: 18 and 50"
        "for ResNet-18 and ResNet-50 respectively.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for finetuning",
    )
    parser.add_argument(
        "--do-envise-eval",
        action="store_true",
        help="Simulate evaluating the model on Lightmatter Envise hardware.",
    )
    parser.add_argument(
        "--do-oob-eval",
        action="store_true",
        help="Simulate evaluating the model on Lightmatter Envise hardware.",
    )
    parser.add_argument(
        "--finetune-with-dft",
        action="store_true",
        help="Fine-tune the model with idiom-ml Differential Fine-tuning (DFT)",
    )
    parser.add_argument(
        "--finetune-with-ept",
        action="store_true",
        help="Fine-tune the model with idiom-ml Envise-Precision Training (EPT)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=False,
        default=None,
        help="Default: None. Checkpoint-path to load "
        "the ResNet-18 model.",
    )
    return parser

