import { useState } from 'react'
import RoleSelector from './components/RoleSelector'
import StudentView from './components/StudentView'
import AdminView from './components/AdminView'

export default function App() {
  const [role, setRole] = useState(null) // null | 'student' | 'admin'

  if (!role) return <RoleSelector onSelect={setRole} />

  return (
    <div className="shell">
      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-logo">
          campus<span>lens</span>
        </div>
        <div className="topbar-role">
          <span className="role-badge" onClick={() => setRole(null)}>
            {role === 'student' ? '🎓 student' : '🔑 admin'} · switch
          </span>
        </div>
      </header>

      {/* Page content */}
      <main className="page-content">
        {role === 'student' ? <StudentView /> : <AdminView />}
      </main>
    </div>
  )
}
