# myapp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from django.urls import include, path

urlpatterns = [
    path('', include('myapp.urls')),
    # ... 其他 URL 模式
]


@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))['data']
        X_new = preprocess(data)
        X_new_vec = vectorizer.transform(X_new)
        y_pred = clf.predict(X_new_vec)
        return JsonResponse({'y_pred': y_pred.tolist()})
    else:
        return JsonResponse({'error': 'Invalid Request'})

# myapp/urls.py
from django.urls import path
from myapp.views import predict

urlpatterns = [
    path('predict', predict, name='predict'),
]

# mysite/urls.py
from django.urls import include, path

urlpatterns = [
    path('', include('myapp.urls')),
]

# mysite/settings.py
INSTALLED_APPS = [
    'myapp',
    # ...
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
