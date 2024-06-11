

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(on d b)
(clear d)
)
(:goal
(and
(on b c)
(on c d))
)
)


