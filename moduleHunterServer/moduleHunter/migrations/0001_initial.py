# Generated by Django 4.0.4 on 2023-03-17 14:26

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModuleInfo',
            fields=[
                ('doc_id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('install', models.CharField(max_length=255)),
                ('repository', models.CharField(max_length=255)),
                ('homepage', models.CharField(max_length=255)),
                ('weekly_downloads', models.IntegerField()),
                ('version', models.CharField(max_length=20)),
                ('license', models.CharField(max_length=100)),
                ('unpacked_size', models.CharField(max_length=20)),
                ('total_files', models.IntegerField()),
                ('issues', models.IntegerField()),
                ('pull_requests', models.IntegerField()),
                ('last_publish', models.CharField(max_length=100)),
                ('url', models.CharField(max_length=255)),
            ],
            options={
                'db_table': 'module_info',
            },
        ),
        migrations.CreateModel(
            name='ModulesDocumentTermFrequency',
            fields=[
                ('doc_id', models.IntegerField(primary_key=True, serialize=False)),
                ('term_frequency', models.JSONField()),
            ],
            options={
                'db_table': 'modules_document_term_frequency',
            },
        ),
        migrations.CreateModel(
            name='ModulesInvertedIndex',
            fields=[
                ('term_id', models.IntegerField(primary_key=True, serialize=False)),
                ('term', models.CharField(max_length=255)),
                ('doc_ids', models.CharField(max_length=255, validators=[django.core.validators.int_list_validator])),
            ],
            options={
                'db_table': 'modules_inverted_index',
            },
        ),
    ]
