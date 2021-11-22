from django.contrib import admin

from .models import Model, Criteria, PortfolioInfo, Stock, Portfolio


class ModelAdmin(admin.ModelAdmin):
    list_display = ("pk", "title")
    search_fields = ("title",)
    empty_value_display = "-пусто-"


class CriteriaAdmin(admin.ModelAdmin):
    list_display = ("pk", "title")
    search_fields = ("title",)
    empty_value_display = "-пусто-"


class PortfolioInfoAdmin(admin.ModelAdmin):
    list_display = ("pk", "title", "criteria", "model")
    search_fields = ("title",)
    empty_value_display = "-пусто-"


class StockAdmin(admin.ModelAdmin):
    list_display = ("pk", "title")
    search_fields = ("title",)
    empty_value_display = "-пусто-"


class PortfolioAdmin(admin.ModelAdmin):
    list_display = ("portfolio", "stock")
    search_fields = ("portfolio", "stock")
    empty_value_display = "-пусто-"


admin.site.register(Model, ModelAdmin)
admin.site.register(Criteria, CriteriaAdmin)
admin.site.register(PortfolioInfo, PortfolioInfoAdmin)
admin.site.register(Stock, StockAdmin)
admin.site.register(Portfolio, PortfolioAdmin)
