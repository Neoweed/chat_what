#!groovy
pipeline {
  agent any
  stages 
    {
    stage('Clean') {
      steps {
        sh 'mvn clean package -DskipTests'
      }
    }
    stage('Build docker images') {
      steps {
        sh 'docker build -t akhilank1937/akhilbuild:1.0 .'
      }
    }
    stage('Docker Hub') {
      steps 
      {
        withDockerRegistry([credentialsId: 'DockerHub', url:""])
        {
          sh 'docker push akhilank1937/akhilbuild:1.0'
        }
      }
    }
  }
}