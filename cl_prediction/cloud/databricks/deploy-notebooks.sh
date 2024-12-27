#!/bin/bash

tenant_id=$1
client_id=$2
client_secret=$3
subscription_id=$4
resourceGroup=$5
workspaceName=$6
notebookPathUnderWorkspace=$7

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

replaceSource="./"
replaceDest=""

find . -type d -name "*" -print0 | while IFS= read -r -d '' dirPath; do
    echo "Processing directory: $dirPath"
    directoryName=${dirPath//$replaceSource/$replaceDest}
    pathOnDatabricks="$notebookPathUnderWorkspace/${directoryName#.}"
    echo "pathOnDatabricks: $pathOnDatabricks"

    JSON="{ \"path\" : \"$pathOnDatabricks\" }"
    curl -X POST https://$workspaceUrl/api/2.0/workspace/mkdirs \
      -H "Authorization:Bearer $accessToken" \
      -H "X-Databricks-Azure-SP-Management-Token: $managementToken" \
      -H "X-Databricks-Azure-Workspace-Resource-Id: $resourceId" \
      -H "Content-Type: application/json" \
      --data "$JSON"
done

find . -type f -name "*" -print0 | while IFS= read -r -d '' file; do
    echo "Processing file: $file"
    filename=${file//$replaceSource/$replaceDest}
    extension="${filename##*.}"

    case $extension in
        sql) language="SQL" ;;
        scala) language="SCALA" ;;
        py) language="PYTHON" ;;
        r) language="R" ;;
        *) language="" ;;
    esac

    if [ -n "$language" ]; then
        curl -n https://$workspaceUrl/api/2.0/workspace/import \
          -H "Authorization:Bearer $accessToken" \
          -H "X-Databricks-Azure-SP-Management-Token: $managementToken" \
          -H "X-Databricks-Azure-Workspace-Resource-Id: $resourceId" \
          -F language="$language" \
          -F overwrite=true \
          -F path="$notebookPathUnderWorkspace/$filename" \
          -F content=@"$file"
    fi
done
