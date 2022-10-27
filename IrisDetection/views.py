from django.shortcuts import render, HttpResponse
from . import mlp_backpropagation
from django.template import loader
# Create your views here.

def index(request):
    return HttpResponse("it works really!")

def evaluation(request, epoch, count, alpha):
    result = {}
    result.clear()
    result = mlp_backpropagation.main(False, Epoch=int(epoch), Hidden_Neuron= int(count), Alpha= float(alpha))

    return render(request, 'IrisDetection/Program.html', result)

def eval(request):
    return render(request, 'IrisDetection/Eval.html')
def start(request):
    return render(request, 'IrisDetection/Start.html')
def test(request):
    return render(request, 'IrisDetection/Test.html')


def tester(request, sepLength, sepWidth, petLength, petWidth):
    result = mlp_backpropagation.main(True, SepalLengthCm= float(sepLength), SepalWidthCm=float(sepWidth)
                                      , PetalLengthCm= float(petLength), PetalWidthCm=float(petWidth))
    flowers = ['Setosa','Versicolor','Verginica']
    context = {
        'prediction': flowers[int(result['predictions'][0])],
        }
    return render(request, 'IrisDetection/Result.html', context)

