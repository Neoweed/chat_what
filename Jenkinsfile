#!groovy
pipeline {
  agent any
  stages 
    {
    stage('Build docker images') {
      steps {
        sh 'docker build -t akhilank1937/chatbuild .'
      }
    }
    stage('Test'){
      steps {
        sh 'python3 test.py' 
      }
    }
    stage('Docker Hub') {
      steps 
      {
        withDockerRegistry([credentialsId: 'DockerHub', url:""])
        {
          sh 'docker push akhilank1937/chatbuild:latest'
        }
      }
    }
    stage('Execute Rundeck job') {
      steps {
  script {
    step([$class: "RundeckNotifier",
          includeRundeckLogs: true,
          jobId: "56b09f91-4460-4e8b-a1b7-ae5d9af86fb6",
          rundeckInstance: "rundeck",
          shouldFailTheBuild: true,
          shouldWaitForRundeckJob: true,
          tags: "",
          tailLog: true])
  }
}
  } 
  }
}