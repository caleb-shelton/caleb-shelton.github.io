import Head from 'next/head'
import { RoughNotation } from "react-rough-notation";
import Prism from "prismjs";
import "prismjs/themes/prism-tomorrow.css";

export default function Home() {
  return (
    <div class="test">
      <Head>
        <title>Caleb Shelton</title>
        <link rel="icon" href="/favicon.ico" />
      </Head> 

      <div className="main">
            <div className="bio">
                <h1>Hello! 👋 I'm
                  <RoughNotation type="highlight" show={true} color="#fff176"> Caleb</RoughNotation>, a Software Engineer <br /> based in Cambridge.</h1>
                <div className="paragraph">
                    I am a Cambridge based
                    <RoughNotation type="underline" animationDelay={1000} show={true} color="#000000"> Software Engineer </RoughNotation> 
                    and
                    <RoughNotation type="underline" animationDelay={2000} show={true} color="#000000"> First Class Graduate </RoughNotation>
                    of Computer Science. <br /> My department awarded me
                    <RoughNotation type="highlight" animationDelay={4000} show={true} color="#9787FF"> 'the best individual performance by a BSc Computing student'.</RoughNotation>
                    <br />
                    I am highly solution focused and excel in problem solving environments. <br />I work
                    full-time as a Software Engineer at Unipart Digital solving complex logistical
                    challenges using Linux, Python, Docker, and cloud technologies.
                </div>
                <div className="buttons">
                    <a href="https://www.linkedin.com/in/caleb-shelton-46a760143/"><button>View LinkedIn</button></a>
                    <a href="https://github.com/caleb-shelton"><button>View Github</button></a>

                </div>
            </div><div className="picture">
            <pre className="line-numbers">
              <code className="language-js">
                {`print("hello world!")`}
              </code>
            </pre>
                {/* <img src="snippet.svg" /> */}
                
            </div>
      </div>

      {/* <div className="main">
            <div className="half">
                <h1>Projects 🚀</h1>
                <div className="paragraph">
                    <h2>Test Project</h2>
                    <i><h3>WIP</h3></i>
                    WIP
                </div>
            </div><div className="half">
                <h1>Skills 💻</h1>
                <div className="paragraph">
                    <ul>
                        <li>Python</li>
                        <li>Linux</li>
                        <li>React.js/Next.js</li>
                        <li>Git</li>
                        <li>Django/Flask/Odoo</li>
                        <li>Cloud development</li>
                        <li>HTML/CSS/UI Design</li>
                        <li>Adobe CC Suite</li>
                        <li>Electronics/Systems Engineering</li>
                        <li>Strong communication and interpersonal skills</li>
                    </ul>
                </div>
            </div>
        </div> */}

      <style jsx global>{`

        * {
          margin: 0;
        }

        body {
          font-family: 'Poppins', sans-serif;

          background-color: rgb(255, 253, 249);
          margin-bottom: 100px;
        }
      `}</style>

      <style jsx>{`
        * {
          margin: 0;
        }
        
        .main {
            width: 60%;
            margin: auto;
            padding-top: 150px;
            line-height: 2;
        }
        
        .bio {
            color: rgb(22, 22, 22);
            width: 70%;
            display: inline-block;
        }
        
        .picture {
            width: 30%;
            display: inline-block;
            vertical-align: top;
            overflow-wrap: break-word;
        }
        
        .half {
            width: 50%;
            display: inline-block;
            vertical-align: top;
            overflow-wrap: break-word;
            font-size: medium;
        }
        
        .paragraph {
            margin-top: 15px;
            font-size: medium;
            padding-right: 20px;
        }
        
        .buttons {
            margin-top: 20px;
        }

        button {
            margin-right: 5px;
        }
        
        h1 {
            line-height: 1.5;
        }
        
        h2 {
            font-weight: 600;
            font-size: medium;
        }
        
        h3 {
            font-weight: 500;
            font-size: medium;
        }
        
        img {
            padding-top: 75px;
            width: 350px;
        }
      `}</style>
    </div>
  )
}
