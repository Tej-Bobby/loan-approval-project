def create_features(X_train, X_test):

    # Create Binary Feature for Coapplicant
    X_train["Has_Coapplicant"] = (
        X_train["CoapplicantIncome"] > 0
    ).astype(int)

    X_test["Has_Coapplicant"] = (
        X_test["CoapplicantIncome"] > 0
    ).astype(int)

    return X_train, X_test