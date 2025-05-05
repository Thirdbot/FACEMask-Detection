import Home from "./components/pages/Home";

if (localStorage.getItem("isExpanded") === null){
  localStorage.setItem("isExpanded", "true");
}

const App = () => {
  return <Home />;
};

export default App;
