import time
from sklearn.metrics import accuracy_score

class pretrainedModel:
    """A class for generating results using a pretrained language model
    Args:
        data_dict (dataset): A dataset containing the train, validation, and test data
        data_selected (str): A string for selecting either the train, validation, or test data
        truncation (bool): A boolean or string for specifying the truncation requirement
        padding (bool): A boolean or string for specifying the padding requirement
        checkpoint (str): A checkpoint for the pretrained model
        tokenizer (obj): A transformer object
        model (obj): A pre-trained model
        device (obj): Specifies whether to use cpu or gpu
    """
    def __init__(self, data_dict, data_selected, truncation, padding, checkpoint, tokenizer, model, device):
        self.data_dict = data_dict
        self.data_selected = data_selected
        self.truncation = truncation
        self.padding = padding
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate_results(self):
        """A method for generating responses based on the input text
        Returns:
            Predictions for the classifers
        """
        start_time = time.time()
        print(f"Generating outputs ...")
        print(f"Model used: {self.checkpoint}")
        preds = []
        for i in range(len(self.data_dict[self.data_selected])):
            # Encode the input sentence
            encoded_inputs = self.tokenizer(self.data_dict[self.data_selected]['features'][i], padding=self.padding, truncation=self.truncation, return_tensors="pt").to(self.device)
            # Generate the predictions
            outputs = self.model.generate(**encoded_inputs)
            # Decode the predictions
            preds.append(self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
        end_time = time.time()
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return preds


    def scoring_metrics(self, preds):
        """Compute and print the scoring metric
        Args:
            preds (list): A list containing the predictions
        """
        preds_int = []
        for i in range(len(self.data_dict[self.data_selected])):
            if self.data_dict[self.data_selected]['labels'][i] == preds[i]:
                preds_int.append(self.data_dict[self.data_selected]['labels_int'][i])
            else:
                if self.data_dict[self.data_selected]['labels_int'][i] == 1:
                    preds_int.append(2)
                else:
                    preds_int.append(1)
        print("Accuracy: ", accuracy_score(self.data_dict[self.data_selected]['labels'], preds))