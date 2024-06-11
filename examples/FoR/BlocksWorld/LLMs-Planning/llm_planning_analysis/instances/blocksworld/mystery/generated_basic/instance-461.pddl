

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a c)
(planet b)
(planet c)
(craves d b)
(craves e d)
(province a)
(province e)
)
(:goal
(and
(craves b a)
(craves d e))
)
)


