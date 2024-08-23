from transformers import pipeline
import time
import random
from sklearn.metrics import accuracy_score

class finetunedModel:
    """A class for generating results using a finetuned model
    Args:
        repository_id (str): A string id for the repository
        device (str): A string for selecting either cpu or gpu
        model_type (str): A string indicating the type of model
        data_dict (dataset): A dataset containing the train, validation, and test data
        data_selected (str): A string for selecting either the train, validation, or test data
        pred_filepath (str): The folder path to save the predictions
    """
    def __init__(self, repository_id, device, model_type, data_dict, data_selected, pred_filepath):
        self.repository_id = repository_id
        self.device = device
        self.model_type = model_type
        self.data_dict = data_dict
        self.data_selected = data_selected
        self.predicted_labels = []
        self.predicted_labels_int = []
        self.input_data_selected = self.data_dict[self.data_selected]
        self.true_labels = self.input_data_selected["labels"]
        self.true_labels_int = self.input_data_selected["labels_int"]
        self.pred_filepath = pred_filepath

    def load_model(self):
        """A method for loading a fine-tuned model
        Returns:
            A fined-tuned model
        """
        # Load model
        start_time = time.time()
        print(f"Loading model...{self.repository_id}")
        loaded_model = pipeline(self.model_type, model=self.repository_id, device=self.device)
        end_time = time.time()
        print("Completed.")
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return loaded_model

    def generate_pred(self, model):
        """Generate the predicted labels
        Args:
            model (obj): A pre-trained model
        Returns:
            A dataframe containing the features, labels, and predicted labels
        """
        start_time = time.time()
        print("Generating predictions...")
        # Prepare input data
        input_data = self.input_data_selected['features']
        # Generate predictions
        for i in range(len(self.input_data_selected['features'])):
            predicted_label = model(input_data[i])[0]['generated_text']
            # Append the binary predicted values for the labels
            if predicted_label == self.input_data_selected['option1'][i]:
                self.predicted_labels_int.append(1)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option2'][i]:
                self.predicted_labels_int.append(2)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option3'][i]:
                self.predicted_labels_int.append(3)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option4'][i]:
                self.predicted_labels_int.append(4)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            elif predicted_label == self.input_data_selected['option5'][i]:
                self.predicted_labels_int.append(5)
                # Append the text predicted labels
                self.predicted_labels.append(model(input_data[i])[0]['generated_text'])
            else:
                rand_int = random.randint(1, 5)
                self.predicted_labels_int.append(rand_int)
                if rand_int == 1:
                    self.predicted_labels.append(self.input_data_selected['option1'][i])
                if rand_int == 2:
                    self.predicted_labels.append(self.input_data_selected['option2'][i])
                if rand_int == 3:
                    self.predicted_labels.append(self.input_data_selected['option3'][i])
                if rand_int == 4:
                    self.predicted_labels.append(self.input_data_selected['option4'][i])
                if rand_int == 5:
                    self.predicted_labels.append(self.input_data_selected['option5'][i])

        # Compile results into a dataframe
        res_df = pd.DataFrame({
            'original_sentence': self.input_data_selected['original_sentence'],
            "features": self.input_data_selected['features'],
            "labels": self.input_data_selected['labels'],
            "labels_int": self.input_data_selected['labels_int'],
            "option1": self.input_data_selected['option1'],
            "option2": self.input_data_selected['option2'],
            "option3": self.input_data_selected['option3'],
            "option4": self.input_data_selected['option4'],
            "option5": self.input_data_selected['option5'],
            "predicted_labels": self.predicted_labels,
            "predicted_labels_int": self.predicted_labels_int
        })
        end_time = time.time()
        print("Completed.")
        print(f"Total time taken: {(end_time-start_time)/60} mins")
        return res_df

    def scoring_metric(self):
        """Generate the accuracy score for the predicted labels
        """
        print("Accuracy: ", accuracy_score(self.true_labels_int, self.predicted_labels_int))

    def save_preds(self, preds):
        """Save predictions to a csv file
        """
        preds['predicted_labels_int'].to_csv(self.pred_filepath + '.csv', index=False, header = False)
        preds.to_excel(self.pred_filepath + '.xlsx', index=False)
        with open(self.pred_filepath + '.txt','w') as f:#, encoding='utf-16-le') as f:
            for p in preds['predicted_labels_int']: f.write(f"{strip(p)}\n")
        print(f"Predictions saved to: {self.pred_filepath}")