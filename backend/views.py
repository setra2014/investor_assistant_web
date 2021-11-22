from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from .portfolio_models import runModel

from .models import *
from .forms import PortfolioForm

from django.views.generic import CreateView
from django.urls import reverse_lazy


def create_portfolio_2(portfolio, deposit, min_klaster, max_klaster, stocks):

    stocks = stocks.split(',')

    print(stocks, int(min_klaster), int(max_klaster), deposit, portfolio.criteria.title, portfolio.model.title)

    # x - список value, сам портфель; criteria - лучшее значение критерия
    x, criteria = runModel(stocks, int(min_klaster), int(max_klaster), deposit, portfolio.criteria.title, portfolio.model.title)

    print(criteria)
    for stock in x:
        Portfolio.objects.create(
            portfolio=portfolio,
            stock=Stock.objects.create(title=stock),
            value=x[stock][0],
            currency=x[stock][0]/100 * int(deposit),
            criteria=criteria
        )

def create_portfolio(request):
    if request.method == "POST":
        form = PortfolioForm(request.POST or None)
        if form.is_valid():
            portfolio = form.save(commit=False)
            portfolio.save()
            deposit = int(form.data['deposit'])
            min_klaster = form.data['min_klaster']
            max_klaster = form.data['max_klaster']
            stocks = form.data['stocks']
            create_portfolio_2(portfolio, deposit, min_klaster, max_klaster, stocks)
            return redirect("get_portfolios")
        return render(request, "new.html", {"form": form})
    form = PortfolioForm()
    return render(request, "new.html", {"form": form})

def get_portfolios(request):
    result = []

    portfolios = PortfolioInfo.objects.all()
    stocks = Portfolio.objects.filter()

    for portfolio in portfolios: 
        stocks_objects = Portfolio.objects.filter(portfolio=portfolio)
        stocks = ''
        best_criteria = ''
        for stock in stocks_objects:
            stocks += f'{stock.stock}: доля депозита: {str(stock.value)}%, в рублях: {stock.currency}\n'
            best_criteria = stock.criteria

        result.append(
            {
                "title": portfolio.title,
                "model": portfolio.model,
                "criteria": f'{portfolio.criteria}: {best_criteria}',
                "stocks": stocks,
            }
        )


    return render(
         request,
         "portfolios.html",
         {
             "portfolios": result,
         }
     )
