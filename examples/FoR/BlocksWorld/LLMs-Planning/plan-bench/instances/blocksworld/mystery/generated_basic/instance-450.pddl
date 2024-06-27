

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a c)
(planet b)
(craves c d)
(craves d b)
(planet e)
(province a)
(province e)
)
(:goal
(and
(craves a c)
(craves d a)
(craves e d))
)
)


