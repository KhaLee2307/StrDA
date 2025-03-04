import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor, VGG_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.stages = {
            "Trans": args.Transformation,
            "Feat": args.FeatureExtraction,
            "Seq": args.SequenceModeling,
            "Pred": args.Prediction,
        }

        """ Transformation """
        if args.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=args.num_fiducial,
                I_size=(args.imgH, args.imgW),
                I_r_size=(args.imgH, args.imgW),
                I_channel_num=args.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if args.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                args.input_channel, args.output_channel
            )
        elif args.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                args.input_channel, args.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")

        self.FeatureExtraction_output = args.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling """
        if args.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output, args.hidden_size, args.hidden_size
                ),
                BidirectionalLSTM(
                    args.hidden_size, args.hidden_size, args.hidden_size
                ),
            )
            self.SequenceModeling_output = args.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if args.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, args.num_class)
        elif args.Prediction == "Attn":
            self.Prediction = Attention(
                self.SequenceModeling_output, args.hidden_size, args.num_class
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, image, text=None, is_train=True):
        """ Transformation stage """
        if not self.stages["Trans"] == "None":
            image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(
            0, 3, 1, 2
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(
            visual_feature
        )  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(
                visual_feature
            )  # [b, num_steps, args.hidden_size]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.args.batch_max_length,
            )

        return prediction  # [b, num_steps, args.num_class]
    

class BaselineClassifier(nn.Module):
    """
    Baseline model for discriminaton method
    
    """

    def __init__(self, args):
        super(BaselineClassifier, self).__init__()
        self.args = args
        
        self.stages = {
            "Trans": args.Transformation,
            "Feat": args.FeatureExtraction,
            "Seq": args.SequenceModeling,
            "Pred": args.Prediction,
        }

        """ Transformation """
        if args.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=args.num_fiducial,
                I_size=(args.imgH, args.imgW),
                I_r_size=(args.imgH, args.imgW),
                I_channel_num=args.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if args.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                args.input_channel, args.output_channel
            )
        elif args.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                args.input_channel, args.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")

        self.FeatureExtraction_output = args.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """ Binary classifier"""
        self.AdaptiveAvgPool_2 = nn.AdaptiveAvgPool2d((None, 1))
        self.Classifier_input = self.FeatureExtraction_output
        self.predict = nn.Linear(self.Classifier_input, 1)

    def forward(self, image, extract_feature = False):
        """ Transformation stage """
        if not self.stages["Trans"] == "None":
            image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(
            0, 3, 1, 2
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(
            visual_feature
        )  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        visual_feature = visual_feature.permute(0, 2, 1)  # [b, w, c] -> [b, c, w]
        visual_feature = self.AdaptiveAvgPool_2(
                visual_feature
            )  # [b, c, w] -> [b, c, 1]
        visual_feature = visual_feature.squeeze(2)  # [b, c, 1] -> [b, c]
        
        """ Binary classifier"""
        output = self.predict(visual_feature) # [b, c] -> [b, class]

        if extract_feature == True:
            return output, visual_feature

        return output
