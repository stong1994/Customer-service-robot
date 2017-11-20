# import test
from django.shortcuts import render
from django.shortcuts import HttpResponse
from templatetags import testapi


# Create your views here.
def index(request):
    # return HttpResponse("hello world")
    return render(request, "index.html")
def ajaxdata(request):
    # 获取用户输入内容
    value = request.POST.get("value", None)
    # 根据用户内容，获取语句类别
    a = testapi.test(value)
    print("value", a)
    return HttpResponse(a)
