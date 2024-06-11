

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a b)
(craves b c)
(planet c)
(craves d a)
(planet e)
(province d)
(province e)
)
(:goal
(and
(craves a c)
(craves d a)
(craves e b))
)
)


