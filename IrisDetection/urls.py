from django.urls import path, re_path
from . import views



urlpatterns = [
    path('', views.index, name = 'index'),
re_path(r'^(?P<sepLength>[0-9]\.[0-9])%(?P<sepWidth>[0-9]\.[0-9])%(?P<petLength>[0-9]\.[0-9])%(?P<petWidth>[0-9]\.[0-9])$', views.tester, name = 'tester'),
    path('<epoch>/<count>/<alpha>', views.evaluation, name = 'evaluation'),
    # path('<sepLength>/<sepWidth>/<petLength>/<petWidth>', views.tester, name = 'tester'),

    path('eval', views.eval, name = "eval"),
    path('start', views.start, name = "start"),
    path('test', views.test, name = "test")
]