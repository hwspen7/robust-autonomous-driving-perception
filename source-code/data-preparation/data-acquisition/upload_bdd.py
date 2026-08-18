from modelscope.hub.api import HubApi

api = HubApi()

api.upload_file(
    path_or_fileobj="bdd100k_coco.tar.gz",
    path_in_repo="bdd100k_coco.tar.gz",
    repo_id="dgadsfgaSFAwsf/autodrive",
    repo_type="dataset",
)

print("Upload finished")

