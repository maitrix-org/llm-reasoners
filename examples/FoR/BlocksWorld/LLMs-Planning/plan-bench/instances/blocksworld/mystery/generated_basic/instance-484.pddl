

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a e)
(planet b)
(craves c b)
(craves d c)
(planet e)
(province a)
(province d)
)
(:goal
(and
(craves b c)
(craves c e)
(craves d a)
(craves e d))
)
)


