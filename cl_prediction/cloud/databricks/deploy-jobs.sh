#!/bin/bash

tenant_id=$1
client_id=$2
client_secret=$3
subscription_id=$4
resourceGroup=$5
workspaceName=$6

azure_databricks_resource_id="2ff814a6-3304-4ab8-85cb-cd0e6f879c1d"
resourceId="/subscriptions/$subscription_id/resourceGroups/$resourceGroup/providers/Microsoft.Databricks/workspaces/$workspaceName"

accessToken=$(curl -X POST https://login.microsoftonline.com/$tenant_id/oauth2/token \
  -F resource=$azure_databricks_resource_id \
  -F client_id=$client_id \
  -F grant_type=client_credentials \
  -F client_secret=$client_secret | jq .access_token --raw-output)

managementToken=$(curl -X POST https://login.microsoftonline.com/$tenant_id/oauth2/token \
  -F resource=https://management.core.windows.net/ \
  -F client_id=$client_id \
  -F grant_type=client_credentials \
  -F client_secret=$client_secret | jq .access_token --raw-output)

workspaceUrl=$(curl -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $managementToken" \
  https://management.azure.com/subscriptions/$subscription_id/resourcegroups/$resourceGroup/providers/Microsoft.Databricks/workspaces/$workspaceName?api-version=2018-04-01 \
  | jq .properties.workspaceUrl --raw-output)

echo "Databricks workspaceUrl: $workspaceUrl"

clusterList=$(curl GET https://$workspaceUrl/api/2.0/clusters/list \
  -H "Authorization:Bearer $accessToken" \
  -H "X-Databricks-Azure-SP-Management-Token: $managementToken" \
  -H "X-Databricks-Azure-Workspace-Resource-Id: $resourceId" \
  -H "Content-Type: application/json")

jobList=$(curl GET https://$workspaceUrl/api/2.0/jobs/list \
  -H "Authorization:Bearer $accessToken" \
  -H "X-Databricks-Azure-SP-Management-Token: $managementToken" \
  -H "X-Databricks-Azure-Workspace-Resource-Id: $resourceId" \
  -H "Content-Type: application/json")

find . -type f -name "*" -print0 | while IFS= read -r -d '' file; do
    echo "Processing file: $file"
    filename=${file//$replaceSource/$replaceDest}
    echo "New filename: $filename"

    jobName=$(cat $filename | jq -r .name)
    jobId=$(echo $jobList | jq -r ".jobs[] | select(.settings.name == \"$jobName\") | .job_id")
    echo "jobName: $jobName"
    echo "jobId: $jobId"

    existing_cluster_id_ClusterName=$(cat $filename | jq -r .existing_cluster_id)
    echo "existing_cluster_id_ClusterName: $existing_cluster_id_ClusterName"
    clusterId=$(echo $clusterList | jq -r ".clusters[] | select(.cluster_name == \"$existing_cluster_id_ClusterName\") | .cluster_id")
    echo "clusterId: $clusterId"

    if [ $existing_cluster_id_ClusterName != "null" ] && [ -z "$clusterId" ]; then
        echo "ERROR: Cluster ($existing_cluster_id_ClusterName) not found."
        exit 1
    fi

    json=$(cat $filename)
    if [ -n "$clusterId" ]; then
        json=$(echo $json | jq -r ".existing_cluster_id = \"$clusterId\"")
    fi

    if [ -z "$jobId" ]; then
        echo "Creating job $jobName..."
        curl -X POST https://$workspaceUrl/api/2.0/jobs/create \
          -H "Authorization:Bearer $accessToken" \
          -H "X-Databricks-Azure-SP-Management-Token: $managementToken" \
          -H "X-Databricks-Azure-Workspace-Resource-Id: $resourceId" \
          -H "Content-Type: application/json" \
          --data "$json"
    else
        echo "Updating job $jobName..."
        json="{ \"job_id\" : $jobId, \"new_settings\": $json }"
        curl -X POST https://$workspaceUrl/api/2.0/jobs/reset \
          -H "Authorization:Bearer $accessToken" \
          -H "X-Databricks-Azure-SP-Management-Token: $managementToken" \
          -H "X-Databricks-Azure-Workspace-Resource-Id: $resourceId" \
          -H "Content-Type: application/json" \
          --data "$json"
    fi
    echo ""
done
