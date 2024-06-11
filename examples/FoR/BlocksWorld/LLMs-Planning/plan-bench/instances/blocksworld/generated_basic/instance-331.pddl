

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(ontable b)
(ontable c)
(on d b)
(clear a)
(clear d)
)
(:goal
(and
(on a d)
(on b a))
)
)


