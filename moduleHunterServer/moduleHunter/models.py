from django.db import models
from django.db.models import JSONField
from django.core.validators import int_list_validator

# Create your models here.

class ModuleInfo(models.Model):
    doc_id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    install = models.CharField(max_length=255)
    repository = models.CharField(max_length=255)
    homepage = models.CharField(max_length=255)
    weekly_downloads = models.IntegerField()
    version = models.CharField(max_length=20)
    license = models.CharField(max_length=100)
    unpacked_size = models.CharField(max_length=20)
    total_files = models.IntegerField()
    issues = models.IntegerField()
    pull_requests = models.IntegerField()
    last_publish = models.CharField(max_length=100)
    url = models.CharField(max_length=255)
    class Meta:
        db_table = 'module_info'

class ModulesDocumentTermFrequency(models.Model):
    doc_id = models.IntegerField(primary_key=True)
    term_frequency = JSONField()
    class Meta:
        db_table = 'modules_document_term_frequency'

class ModulesInvertedIndex(models.Model):
    term_id = models.IntegerField(primary_key=True)
    term = models.CharField(max_length=255)
    doc_ids = models.CharField(validators=[int_list_validator], max_length=255) 
    class Meta:
        db_table = 'modules_inverted_index'