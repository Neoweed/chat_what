#!groovy
pipeline {
  agent any
  stages 
    {
    stage('Build docker images') {
      steps {
        sh 'docker build -t akhilank1937/chatbuild:1.0 .'
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
          rundeckInstance: "rundeck",
          jobId: "56b09f91-4460-4e8b-a1b7-ae5d9af86fb6",
          shouldFailTheBuild: true,
          shouldWaitForRundeckJob: true,
          tailLog: true])
     }
   }
  } 
  }
}