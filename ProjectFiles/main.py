'''
Important!
Enter your full name (as it appears on Canvas) and NetID.  
If you are working in a group (maximum of 3 members), include the full names and NetIDs of all your partners.  
If you're working alone, enter `None` for the partner fields.
'''

'''
Project: MP8
Student 1: Ashley Williams, adwilliams7
Student 2: None
Student 3: None
'''

# Add imports used throughout the project here
# import pandas as pd # for main
# import os # for main
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # for debug

lr = None

class UserPredictor:
    def fit(self, train_user, train_logs, train_y): 
        # given data from train csvs, create a 'sklearn` model to fit the data
        # using the train data, create a lr to predict if the user clicked the email or not (y data)
        # use features: total seconds spent looking and total purchace amount to predict *add additional features if needed
        global lr
        lr = None
        
        features_df = self.create_df(train_user, train_logs)
        features = ["seconds", "past_purchase_amt", "age", "badge"]
        
        x_train, x_test, y_train, y_test = train_test_split(features_df[features], train_y["y"], test_size=0.2)
                
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        
        # debug, maybe delete after done
        # scores = cross_val_score(lr, features_df[features], train_y["y"])
        # print(f"AVG: {scores.mean()}, STD: {scores.std()}\n")


    def predict(self, test_user, test_logs):
        global lr
        if lr: 
            features_df = self.create_df(test_user, test_logs)
            features = ["seconds", "past_purchase_amt", "age", "badge"]
                        
            return lr.predict(features_df[features])
        else: 
            return None


    def create_df(self, user, logs):
        # get total time spent by each user (where data is available)
        time_df = logs.groupby("user_id")["seconds"].sum()
        
        # merge time_df into users_df, using user_id; fill nan with 0
        features_df = user.merge(time_df, how="left", on="user_id") 
        features_df.fillna(0, inplace=True)
        
        # turn badges into numerical values
        badges = ["bronze", "silver", "gold"]
        for idx in range(len(badges)):
            features_df["badge"] = features_df["badge"].replace(badges[idx], idx)
        
        return features_df
        
    
    
# TESTING PURPOSES ONLY : DELETE ONCE DONE
# def main():
#     predictor = UserPredictor()
#     train_users = pd.read_csv(os.path.join("data", "train_users.csv"))
#     train_logs = pd.read_csv(os.path.join("data", "train_logs.csv"))
#     train_y = pd.read_csv(os.path.join("data", "train_y.csv"))
#     test_users = pd.read_csv(os.path.join("data", "test1_users.csv"))
#     test_logs = pd.read_csv(os.path.join("data", "test1_logs.csv"))
    
#     predictor.fit(train_users, train_logs, train_y)
#     print(predictor.predict(test_users, test_logs))
    
# if __name__ == "__main__":
#     main()

# Put the following in 'fit' as a debug print statement
#scores = cross_val_score(model, train_users[self.xcols], train_y["y"])
#print(f"AVG: {scores.mean()}, STD: {scores.std()}\n")

# read tester again at the end to make sure u are doing cross stuff