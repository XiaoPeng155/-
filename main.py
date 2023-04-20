import jieba
import sklearn
import joblib
import sklearn
from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer
import sklearn
from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble


# 加载自定义词典
jieba.load_userdict('/Users/pengjianhui/Desktop/templates/userdict.txt')

# 读取数据
docs = [
    '尊敬的客户，您的信用卡账户出现异常，是否需要冻结该账户？',
    '恭喜您获得XX公司人民币10万元大奖，请尽快回电领取。',
    '亲爱的用户，您好，最近发现您的账户存在违规操作，请及时更改密码。',
    '你看那部电影了吗？',
    '明天我们要去游泳池玩水，你想不想参加呢？'
]

# 分词、统计词频
def preprocess(docs):
    X = []
    for doc in docs:
        tokens = list(jieba.cut(doc)) # 切分文本中的词语
        freqs = {} # 统计词频
        for token in tokens:
            if token not in freqs:
                freqs[token] = 1
            else:
                freqs[token] += 1 
        X.append(freqs)
    return X

X_train = preprocess(docs)

# 定义标签，1表示涉及诈骗金额，0表示其他情况
y_train = [1, 1, 1, 0, 0]

# 构建模型训练集
vectorizer = sklearn.feature_extraction.DictVectorizer()
clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
X_train_vec = vectorizer.fit_transform(X_train)

# 训练模型
clf.fit(X_train_vec, y_train)

# 保存模型
joblib.dump(clf, "/Users/pengjianhui/Desktop/templates/model.pkl")

# 加载模型
clf = joblib.load('/Users/pengjianhui/Desktop/templates/model.pkl')

# 测试新数据
new_docs = ['你好，我在网上购物花了三百元，请问下单成功了吗？']
X_new = preprocess(new_docs)
X_new_vec = vectorizer.transform(X_new)
y_pred = clf.predict(X_new_vec)
print(y_pred)

# 输出：[0]
