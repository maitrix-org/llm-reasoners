

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a b)
(craves b d)
(planet c)
(planet d)
(province a)
(province c)
)
(:goal
(and
(craves a d)
(craves b a))
)
)


