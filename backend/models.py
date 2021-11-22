from django.db import models


class Model(models.Model):
    title = models.CharField(max_length=200, unique=True)

    def __str__(self) -> str:
        return self.title


class Criteria(models.Model):
    title = models.CharField(max_length=200, unique=True)

    def __str__(self) -> str:
        return self.title


class PortfolioInfo(models.Model):
    title = models.CharField(max_length=200)
    criteria = models.ForeignKey(
        Criteria,
        on_delete=models.CASCADE,
    )
    model = models.ForeignKey(
        Model,
        on_delete=models.CASCADE,
    )


class Stock(models.Model):
    title = models.CharField(max_length=200, unique=True)

    def __str__(self) -> str:
        return self.title


class Portfolio(models.Model):
    portfolio = models.ForeignKey(
        PortfolioInfo,
        on_delete=models.CASCADE,
    )
    stock = models.ForeignKey(
        Stock,
        on_delete=models.CASCADE,
    )
    value = models.FloatField(null=True)
    currency = models.FloatField(null=True)
    criteria = models.CharField(max_length=200)
