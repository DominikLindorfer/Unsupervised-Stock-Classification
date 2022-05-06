from django.shortcuts import render, redirect
from .forms import DweetForm
from .models import Dweet, MLResult, MLResultBlueChips

# # Create your views here.
# def dashboard(request):
#     return render(request, "base.html")

# def dashboard(request):
#     form = DweetForm()
#     return render(request, "bluedivis/dashboard.html", {"form": form})

def dashboard(request):

    if request.method == "POST":
        form = DweetForm(request.POST)
        if form.is_valid():
            dweet = form.save(commit=False)
            dweet.user = request.user
            if "divis" in request.POST:
                dweet.is_champion = True
            elif "bluechips" in request.POST:
                dweet.is_bluechip = True
            print("Set Champion: ", dweet.is_champion, dweet.is_bluechip, dweet.user)

            dweet.save()
            return redirect("bluedivis:dashboard")
            
    form = DweetForm()
    dweets = Dweet.objects.all()
    dweets = list(reversed(list(dweets)[-5:]))

    for id in range(len(dweets)):
        dweets[id].id = id
    
    result = MLResult.objects.all()
    result_bluechips = MLResultBlueChips.objects.all()

    for dweet in dweets:
        dweet.cluster = 99
        dweet.set_url()
        if dweet.is_champion:
            for res in result:
                if dweet.body == res.TICKER:
                    dweet.cluster = res.cluster
                    # print(res.TICKER, res.cluster, dweet.cluster)
                    break
                    # print(dweet.body, res.TICKER, dweet.cluster)
        if dweet.is_bluechip:
            dweet.cluster = 99
            dweet.set_url()
            for res in result_bluechips:
                if dweet.body == res.TICKER:
                    dweet.cluster = res.cluster
                    # print(res.TICKER, res.cluster, dweet.cluster)
                    break
                    # print(dweet.body, res.TICKER, dweet.cluster)

    return render(request, "bluedivis/dashboard.html", {"form": form, "dweets": dweets})


def profile_list(request):
    profiles = Dweet.objects.exclude() 
    return render(request, "bluedivis/profile_list.html", {"profiles": profiles})