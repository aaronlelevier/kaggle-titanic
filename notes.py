pdf = pd.read_csv('test.csv')

ple_sex = LabelEncoder().fit_transform(pdf['Sex'])
ple_age = StandardScaler().fit_transform(pdf['Age'].fillna(0)[:, None])
ple_fare = StandardScaler().fit_transform(pdf['Fare'].fillna(0)[:, None])
ple_cabin = LabelEncoder().fit_transform(pdf['Cabin'].fillna(''))
ple_embarked = LabelEncoder().fit_transform(pdf['Embarked'].fillna(''))

ple_age = np.squeeze(ple_age)
ple_fare = np.squeeze(ple_fare)

all_pdf = np.array([ple_sex, ple_age, ple_fare, ple_cabin, ple_embarked ,pdf['Pclass'], pdf['SibSp'], pdf['Parch']])

print(all_pdf.shape)