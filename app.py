from flask import Flask, request, render_template
import jieba
import joblib
from sklearn.feature_extraction import DictVectorizer


app = Flask(__name__)




# 初始化特征向量器


vectorizer = DictVectorizer()

# 加载自定义词典
jieba.load_userdict('/Users/pengjianhui/Desktop/templates/userdict.txt')

# 载入训练好的模型
clf = joblib.load('/Users/pengjianhui/Desktop/templates/model.pkl')

# 创建Flask应用实例
app = Flask(__name__)

# 处理GET请求，返回前端页面
@app.route('/', methods=['GET'])
def index():
    return render_template('/Users/pengjianhui/Desktop/templates/index.html')

# 处理POST请求，接收文本并调用模型进行分类
@app.route('/', methods=['POST'])
def classify():
    text = request.form['text']
    X_new = preprocess([text])
    X_new_vec = vectorizer.transform(X_new)
    y_pred = clf.predict(X_new_vec[0])
    result = '垃圾短信' if y_pred[0] == 1 else '非垃圾短信'
    return render_template('index.html', result=result)

# 预处理函数，对文本进行分词并提取词频特征
def preprocess(docs):
    X = []
    for doc in docs:
        tokens = list(jieba.cut(doc))
        freqs = {}
        for token in tokens:
            if token not in freqs:
                freqs[token] = 1
            else:
                freqs[token] += 1
        X.append(freqs)
    return X


if __name__ == '__main__':
   app.run(debug=False)

