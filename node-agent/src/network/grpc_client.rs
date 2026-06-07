#![allow(dead_code)]
use tonic::Request;

use super::arcflare as pb;
use pb::{
    orchestrator_client::OrchestratorClient,
    NodeStatus, PartitionRequest, PartitionAssignment,
};

pub struct OrchestratorConnection {
    client: OrchestratorClient<tonic::transport::Channel>,
}

impl OrchestratorConnection {
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let client = OrchestratorClient::connect(format!("http://{}", addr)).await?;
        Ok(Self { client })
    }

    pub async fn report_status(&mut self, status: NodeStatus) -> Result<(), tonic::Status> {
        self.client.report_node_status(Request::new(status)).await?;
        Ok(())
    }

    pub async fn request_partition(
        &mut self,
        request: PartitionRequest,
    ) -> Result<PartitionAssignment, tonic::Status> {
        let resp = self.client.request_partition(Request::new(request)).await?;
        Ok(resp.into_inner())
    }
}
